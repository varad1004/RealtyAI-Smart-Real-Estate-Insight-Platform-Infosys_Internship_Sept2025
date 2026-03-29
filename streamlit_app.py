import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st
import io
import base64
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from textwrap import dedent
try:
    # Prefer keras 3 API if available
    from keras.models import load_model as keras_load_model
except Exception:  # pragma: no cover
    keras_load_model = None
import tensorflow as tf

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
ASSETS_DIR = BASE_DIR / 'assets'
SEGMENTATION_MODEL_PATH = BASE_DIR / 'best_model.h5'

# App-wide config
st.set_page_config(page_title='RealtyAI Smart Real Estate Insight Platform', layout='wide')

# ---------- Global Theming (CSS) ----------
st.markdown(
    """
    <style>
    :root {
        --grad-start: #4f46e5; /* indigo-600 */
        --grad-end: #9333ea;   /* purple-600 */
        --glass-bg: rgba(255, 255, 255, 0.07);
        --glass-border: rgba(255, 255, 255, 0.18);
        --text-primary: #e5e7eb; /* gray-200 */
        --text-muted: #cbd5e1;   /* gray-300 */
        --shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 35%, #111827 100%);
        color: var(--text-primary);
    }

    /* Top navigation wrapper */
    .top-nav {
        position: relative;
        width: 100%;
        background: linear-gradient(90deg, var(--grad-start), var(--grad-end));
        padding: 14px 18px;
        border-radius: 16px;
        box-shadow: var(--shadow);
        margin-bottom: 14px;
    }

    /* Align nav buttons row */
    .top-nav-row { display: flex; gap: 8px; }

    /* Glassmorphism buttons (global) */
    .stButton > button {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        color: var(--text-primary) !important;
        border-radius: 12px !important;
        padding: 0.55rem 0.9rem !important;
        transition: transform .15s ease, box-shadow .2s ease, border-color .2s ease !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 12px 28px rgba(0,0,0,0.24);
        border-color: rgba(255,255,255,0.35) !important;
    }
    /* Primary button (active nav) */
    .stButton > button[kind="primary"] {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.45) !important;
        box-shadow: 0 14px 30px rgba(147, 51, 234, 0.35) !important;
    }

    /* Glass cards */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 16px 16px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform .2s ease, box-shadow .25s ease, border-color .2s ease;
    }
    .glass-card:hover { transform: translateY(-2px); box-shadow: 0 16px 38px rgba(0,0,0,0.28); border-color: rgba(255,255,255,0.32); }
    .muted { color: var(--text-muted); font-size: 0.92rem; }

    /* Features grid helpers */
    .features-title { margin: 0 0 8px 0; font-weight: 600; }
    .equal-card { min-height: 160px; display: flex; flex-direction: column; justify-content: space-between; }

    /* Footer */
    .app-footer {
        width: 100%;
        margin-top: 24px;
        padding: 12px 16px;
        text-align: center;
        color: var(--text-muted);
        border-top: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(90deg, rgba(79,70,229,0.12), rgba(147,51,234,0.12));
        box-shadow: 0 -10px 30px rgba(147, 51, 234, 0.15) inset;
        border-radius: 12px;
    }

    /* Responsive tweaks */
    @media (max-width: 900px) {
        .equal-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Shared utilities ----------


def _latest(glob_pattern: str, base: Path) -> Path | None:
    files = list(base.glob(glob_pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _data_uri_from_file(path: Path) -> str | None:
    try:
        data = path.read_bytes()
        mime = 'image/svg+xml' if path.suffix.lower() == '.svg' else 'image/png'
        return f"data:{mime};base64," + base64.b64encode(data).decode('utf-8')
    except Exception:
        return None


def load_feature_columns() -> list[str]:
    """Load feature column names for time-series model from artifacts directory.
    Prefers timestamped files, falls back to artifacts/feature_columns.json.
    """
    latest = _latest('feature_columns_*.json', ARTIFACTS_DIR)
    fallback = ARTIFACTS_DIR / 'feature_columns.json'
    path = latest if latest and latest.exists() else (fallback if fallback.exists() else None)
    if not path:
        st.error('No feature_columns JSON found in artifacts/.')
        st.stop()
    with open(path, 'r') as f:
        data = json.load(f)
    cols = data.get('feature_columns') if isinstance(data, dict) else data
    if not isinstance(cols, (list, tuple)):
        st.error('feature_columns JSON has unexpected format.')
        st.stop()
    return list(cols)


def load_artifacts_for_target(target: str):
    """Load latest XGBoost Booster and matching scaler for a given target."""
    model_json = _latest(f'xgb_city_{target}_*.json', MODELS_DIR)
    scaler_path = _latest(f'scaler_{target}_*.joblib', ARTIFACTS_DIR)
    if not model_json or not model_json.exists():
        st.error(f'No model JSON found for target {target} under models/.')
        st.stop()
    if not scaler_path or not scaler_path.exists():
        st.error(f'No scaler artifact found for target {target} under artifacts/.')
        st.stop()
    booster = xgb.Booster()
    booster.load_model(str(model_json))
    scaler = joblib.load(scaler_path)
    return booster, scaler, model_json, scaler_path


# ---------- Top Navigation ----------

def render_top_nav():
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    st.markdown('<div class="top-nav"></div>', unsafe_allow_html=True)
    nav_cols = st.columns([1, 1, 1, 1, 2])

    def nav_btn(label: str, name: str):
        active = st.session_state.page == name
        if active:
            nav_cols[nav_btn.idx].button(label, key=f"top_{name}_active", use_container_width=True, type="primary")
        else:
            if nav_cols[nav_btn.idx].button(label, key=f"top_{name}", use_container_width=True):
                st.session_state.page = name
    nav_btn.idx = 0

    nav_btn("Home", "Home"); nav_btn.idx += 1
    nav_btn("Features", "Features"); nav_btn.idx += 1
    nav_btn("About", "About"); nav_btn.idx += 1
    nav_btn("Contact", "Contact"); nav_btn.idx += 1
    # Spacer column to keep right-aligned space
def load_segmentation_model():
    """Load the Keras/TensorFlow segmentation model from best_model.h5 (cached)."""
    if not SEGMENTATION_MODEL_PATH.exists():
        return None
    # Try keras.load_model first, then tf.keras.load_model
    try:
        if keras_load_model is not None:
            return keras_load_model(str(SEGMENTATION_MODEL_PATH))
    except Exception:
        pass
    try:
        return tf.keras.models.load_model(str(SEGMENTATION_MODEL_PATH))
    except Exception as e:
        st.error(f"Could not load segmentation model: {e}")
        return None


def _preprocess_image(pil_img: Image.Image, target_size=(512, 512)) -> tuple[np.ndarray, np.ndarray]:
    """Return (img_array_batched[1,H,W,3], original_array[H,W,3]) normalized 0..1."""
    img = pil_img.convert('RGB').resize(target_size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0), arr


def _normalize_true_mask(pil_mask: Image.Image | None, target_size=(512, 512)) -> np.ndarray | None:
    """Convert true mask PIL image into class indices array (H,W) with values in {0,1,2}."""
    if pil_mask is None:
        return None
    mk = pil_mask.convert('L').resize(target_size, Image.NEAREST)
    arr = np.asarray(mk).astype(np.uint8)
    # If mask uses 0 and 255 (binary), map 255->1
    if np.max(arr) > 2:
        out = np.zeros_like(arr, dtype=np.uint8)
        out[arr == 255] = 1
        # If there are any other non-zero values, treat as class 2
        other = (arr != 0) & (arr != 255)
        if np.any(other):
            out[other] = 2
        return out
    # Already class-indexed
    return arr.astype(np.uint8)


def _predict_mask(model, img_batched: np.ndarray) -> np.ndarray:
    """Return predicted class map (H,W) uint8 from model output."""
    pred = model.predict(img_batched, verbose=0)
    if pred.ndim == 4 and pred.shape[-1] > 1:
        cls = np.argmax(pred, axis=-1)[0].astype(np.uint8)
    else:
        # Binary output -> threshold to 0/1
        p = pred[0]
        if p.ndim == 3 and p.shape[-1] == 1:
            p = p[..., 0]
        cls = (p > 0.5).astype(np.uint8)
    return cls


def _colorize_mask(mask_hw: np.ndarray) -> np.ndarray:
    """Map classes to RGB colors as float 0..1 array (H,W,3)."""
    colors = {
        0: np.array([0, 0, 0], dtype=np.float32),      # Background
        1: np.array([255, 0, 0], dtype=np.float32),    # Residential
        2: np.array([0, 255, 0], dtype=np.float32),    # Commercial
    }
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for cls, col in colors.items():
        out[mask_hw == cls] = col / 255.0
    return out


def _compute_iou_dice(y_true_hw: np.ndarray | None, y_pred_hw: np.ndarray) -> tuple[float | None, float | None]:
    if y_true_hw is None:
        return None, None
    y_true = y_true_hw.flatten()
    y_pred = y_pred_hw.flatten()
    iou = jaccard_score(y_true, y_pred, labels=[1, 2], average='macro', zero_division=0)
    dice = f1_score(y_true, y_pred, labels=[1, 2], average='macro', zero_division=0)
    return float(iou), float(dice)


def classify_landuse(mask_hw: np.ndarray) -> str:
    """Classify predicted mask as 'Residential' or 'Commercial' by majority area.
    Ties default to Residential. Background (0) pixels are ignored in comparison.
    """
    res = int(np.sum(mask_hw == 1))
    com = int(np.sum(mask_hw == 2))
    return 'Residential' if res >= com else 'Commercial'


def _run_segmentation(model, pil_img: Image.Image, pil_mask: Image.Image | None, size=(512, 512)):
    x_b, x_vis = _preprocess_image(pil_img, size)
    true_mask = _normalize_true_mask(pil_mask, size)
    pred_cls = _predict_mask(model, x_b)
    pred_col = _colorize_mask(pred_cls)
    overlay = np.clip(0.4 * x_vis + 0.6 * pred_col, 0, 1)
    iou, dice = _compute_iou_dice(true_mask, pred_cls)
    return {
        'original': x_vis,
        'pred_mask': pred_cls,
        'pred_colored': pred_col,
        'overlay': overlay,
        'iou': iou,
        'dice': dice,
    }


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out


def preprocess_input(df_in: pd.DataFrame, feature_cols: list[str], scaler) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = df_in.copy()
    # Add missing feature columns as zeros
    missing_cols = [c for c in feature_cols if c not in df.columns]
    for c in missing_cols:
        df[c] = 0.0
    # Keep only feature columns in correct order
    X = df[feature_cols].copy()
    # Ensure numeric
    X = coerce_numeric(X, feature_cols)
    # Fill NaNs: prefer scaler means if available, else 0.0
    if hasattr(scaler, 'mean_') and len(getattr(scaler, 'mean_', [])) == len(feature_cols):
        means = pd.Series(scaler.mean_, index=feature_cols)
        X = X.fillna(means)
    X = X.fillna(0.0)
    # Scale
    X_scaled = scaler.transform(X.values)
    return X, X_scaled, missing_cols


def format_target_label(target: str) -> str:
    # Only one target; keep label simple
    return 'Average predicted price'


# ---------- Page renderers ----------

def home_page():
    # Build a hero section with a custom SVG background
    hero_path = ASSETS_DIR / 'realtyai_hero.svg'
    bg_uri = _data_uri_from_file(hero_path)
    if bg_uri:
        st.markdown(
            f"""
            <style>
            .hero-bg {{
                background-image: url('{bg_uri}');
                background-size: cover;
                background-position: center;
                border-radius: 16px;
                border: 1px solid var(--glass-border);
                box-shadow: var(--shadow);
                padding: 56px 36px;
            }}
            .hero-title {{ margin: 0; }}
            .hero-sub {{ color: var(--text-muted); margin-top: 8px; }}
            </style>
            <div class="hero-bg">
              <h1 class="hero-title">RealtyAi : Smart Real Estate Insight Platform</h1>
              <div class="hero-sub">AI-powered pricing, segmentation, and forecasting—built for buyers, investors, and planners.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="glass-card" style="padding: 28px 30px;">
              <h1 style="margin:0;">RealtyAi : Smart Real Estate Insight Platform</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )


def house_price_page():
    st.header("House price prediction")
    st.caption("Upload a CSV of listing features to estimate sale prices using the latest XGBoost model.")

    # Helpers specific to house price model
    def _latest_saleprice_model() -> Path | None:
        # Prefer tuned/log models if present
        candidates = [
            _latest('xgb_saleprice_log_tuned.pkl', MODELS_DIR),
            _latest('xgb_saleprice_log.pkl', MODELS_DIR),
            _latest('xgb_saleprice.pkl', MODELS_DIR),
        ]
        candidates = [c for c in candidates if c is not None and c.exists()]
        if candidates:
            # If multiple exist, choose most recent by mtime
            return max(candidates, key=os.path.getmtime)
        # As a fallback, pick any pkl matching pattern
        any_pkl = _latest('xgb_saleprice*.pkl', MODELS_DIR)
        return any_pkl if any_pkl and any_pkl.exists() else None

    def load_saleprice_model():
        model_path = _latest_saleprice_model()
        if not model_path:
            st.error("No house price model found under models/ (expected xgb_saleprice*.pkl)")
            st.stop()
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()
        # Do NOT auto-apply any external target scaler for house prices —
        # these can mismatch and explode values. We only apply log1p inverse
        # based on model filename.
        target_scaler = None
        return model, model_path, target_scaler

    def infer_house_feature_columns(model) -> list[str] | None:
        # 1) Try sklearn attribute
        cols = getattr(model, 'feature_names_in_', None)
        if cols is not None:
            return list(cols)
        # 2) Try processed_data/X_train.csv
        xtrain = BASE_DIR / 'processed_data' / 'X_train.csv'
        if xtrain.exists():
            try:
                cols = list(pd.read_csv(xtrain, nrows=1).columns)
                return cols
            except Exception:
                pass
        return None

    def preprocess_house_input(df_in: pd.DataFrame, feature_cols: list[str] | None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        df = df_in.copy()
        # Choose columns: if we don't know features, default to numeric columns except obvious targets
        if not feature_cols:
            blocked = {c.lower() for c in df.columns}
            maybe_targets = {"saleprice", "sale_price", "log_saleprice", "target"}
            use_cols = [c for c in df.columns if (c.lower() not in maybe_targets)]
            X = df[use_cols].select_dtypes(include=[np.number]).copy()
            missing = []
            return X, X.values, missing
        # Known schema path
        missing_cols = [c for c in feature_cols if c not in df.columns]
        for c in missing_cols:
            df[c] = 0.0
        X = df[feature_cols].copy()
        X = coerce_numeric(X, feature_cols).fillna(0.0)
        return X, X.values, missing_cols

    # Load model on-demand
    model, model_path, target_scaler = load_saleprice_model()
    feature_cols = infer_house_feature_columns(model)

   
    uploaded = st.file_uploader('Upload CSV with feature columns', type=['csv'], key='hp_csv')

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Could not read CSV: {e}')
            st.stop()

        st.subheader('Uploaded data')
        st.dataframe(df_in.head(200), use_container_width=True)

        X_df, X_np, missing_cols = preprocess_house_input(df_in, feature_cols)

        if missing_cols:
            st.warning(f'{len(missing_cols)} expected feature columns were missing and filled with 0. Example: {missing_cols[:8]}')

        if st.button('Predict house prices', use_container_width=True):
            with st.spinner('Predicting...'):
                try:
                    # Important: pass a DataFrame so pipelines with column selectors work
                    preds = model.predict(X_df)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                # Inverse transform policy:
                # - If model filename contains 'log', apply expm1 to bring back to price space.
                # - Otherwise, use raw predictions. Avoid generic target scalers to prevent blow-ups.
                final_preds = preds
                used_inverse = None
                if 'log' in str(model_path.name).lower():
                    try:
                        final_preds = np.expm1(preds)
                        used_inverse = 'np.expm1 (log1p inverse)'
                    except Exception:
                        used_inverse = None

            st.success('Prediction complete.')
            if used_inverse:
                st.caption(f"Applied inverse transform: {used_inverse}")

            avg_pred = float(np.mean(final_preds)) if np.size(final_preds) else float('nan')
            st.metric('Average predicted sale price', f"{avg_pred:,.2f}")

            df_out = df_in.copy()
            df_out['predicted_saleprice'] = final_preds
            st.subheader('Preview with predictions')
            st.dataframe(df_out.head(200), use_container_width=True)

            csv_bytes = df_out.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download predictions CSV',
                data=csv_bytes,
                file_name='predictions_saleprice.csv',
                mime='text/csv'
            )
    else:
        st.info('Upload a CSV to begin. If the saved model schema is unavailable, numeric columns in your file will be used.')


def time_series_page():
    st.header("Time series forecasting: ZHVI (city-level)")
    st.caption("Upload a CSV of city features and predict next-month ZHVI using the latest trained model.")

    selected_target = 'ZHVI_AllHomes'  # fixed target per current artifacts

    # Load artifacts on-demand (only in this tab)
    feature_cols = load_feature_columns()
    booster, scaler, model_json_path, scaler_path = load_artifacts_for_target(selected_target)


    uploaded = st.file_uploader('Upload CSV with feature columns', type=['csv'], key='ts_csv')

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Could not read CSV: {e}')
            st.stop()

        st.subheader('Uploaded data ')
        st.dataframe(df_in.head(200), use_container_width=True)

        # Preprocess and show any missing columns
        X_df, X_scaled, missing_cols = preprocess_input(df_in, feature_cols, scaler)

        if missing_cols:
            st.warning(f'{len(missing_cols)} expected feature columns were missing and filled with 0. Example: {missing_cols[:8]}')

        if st.button('Predict', use_container_width=True):
            with st.spinner('Predicting...'):
                dmat = xgb.DMatrix(X_scaled)
                preds = booster.predict(dmat)
                # Prepare output
                colname = f'predicted_{selected_target}'
                df_out = df_in.copy()
                df_out[colname] = preds
                avg_pred = float(np.mean(preds)) if preds.size else float('nan')

            st.success('Prediction complete.')
            st.metric(format_target_label(selected_target), f"{avg_pred:,.2f}")

            st.subheader('Preview with predictions ')
            st.dataframe(df_out.head(200), use_container_width=True)

            # Detailed time-series chart of predictions
            st.subheader('Predicted ZHVI over time')
            date_col = next((c for c in df_out.columns if c.lower() == 'date'), None)
            plot_df = df_out.copy()
            id_candidates = ['RegionName', 'City', 'Metro', 'CountyName', 'StateName', 'RegionID']
            entity_col = next((c for c in id_candidates if c in plot_df.columns), None)

            # Resolve date column if present
            if date_col is not None:
                plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[date_col])
                plot_df = plot_df.sort_values(by=date_col)

            # Chart options
            with st.expander('Chart options', expanded=True):
                if entity_col is not None and plot_df[entity_col].nunique() > 1:
                    view_mode = st.radio('View', ['Single entity', 'Compare multiple'], horizontal=True, key='ts_view')
                    if view_mode == 'Single entity':
                        chosen_entity = st.selectbox(
                            'Select entity to display',
                            sorted(plot_df[entity_col].dropna().unique().tolist()),
                            key='ts_single_entity'
                        )
                        plot_df_use = plot_df[plot_df[entity_col] == chosen_entity].copy()
                    else:
                        options = sorted(plot_df[entity_col].dropna().unique().tolist())
                        default_sel = options[:5]
                        chosen_multi = st.multiselect('Select up to 8 entities to compare', options, default=default_sel, key='ts_multi')
                        if len(chosen_multi) > 8:
                            st.warning('Please select at most 8 entities for readability.')
                            chosen_multi = chosen_multi[:8]
                        plot_df_use = plot_df[plot_df[entity_col].isin(chosen_multi)].copy()
                else:
                    plot_df_use = plot_df.copy()

                 

            # Build Plotly figure
            if date_col is not None and not plot_df_use.empty:
                if entity_col is not None and plot_df_use[entity_col].nunique() > 1:
                    fig = px.line(
                        plot_df_use,
                        x=date_col,
                        y=colname,
                        color=entity_col,
                        markers=True,
                        labels={date_col: 'Date', colname: 'Predicted ZHVI'},
                        title=None,
                    )
                else:
                    fig = px.line(
                        plot_df_use,
                        x=date_col,
                        y=colname,
                        markers=True,
                        labels={date_col: 'Date', colname: 'Predicted ZHVI'},
                        title=None,
                    )

                # Optional moving average overlay
                
        

                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Predicted ZHVI',
                    xaxis_rangeslider_visible=True,
                    legend_title_text=entity_col if (entity_col is not None and plot_df_use[entity_col].nunique() > 1) else None,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                fig.update_yaxes(tickformat=",.")
                st.plotly_chart(fig, use_container_width=True)
                st.caption('Interactive chart: pan, zoom, and use the range slider to focus on specific periods.')
            else:
                # Fallback — no Date column detected
                plot_df_use = plot_df_use.reset_index(drop=True)
                plot_df_use['Index'] = plot_df_use.index + 1
                fig = px.line(
                    plot_df_use,
                    x='Index',
                    y=colname,
                    color=entity_col if (entity_col is not None and plot_df_use[entity_col].nunique() > 1) else None,
                    markers=True,
                    labels={'Index': 'Row', colname: 'Predicted ZHVI'},
                    title=None,
                )
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                fig.update_yaxes(tickformat=",.")
                st.plotly_chart(fig, use_container_width=True)
                st.caption('No Date column detected. Displaying predictions by row order.')

            # Download button
            csv_bytes = df_out.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download predictions CSV',
                data=csv_bytes,
                file_name=f'predictions_{selected_target}.csv',
                mime='text/csv'
            )
    


def spacenet_page():
    st.header("Spacenet imagery preview")
    st.caption("Upload a tile, run the segmentation model, and download the results.")

    # (Sample tiles gallery removed as requested)
    img_dir = BASE_DIR / 'cleaned' / 'images'  # kept for potential future use

    st.markdown("---")
    st.subheader("Upload image and predict")

    model = load_segmentation_model()
    if model is None:
        st.warning("Place best_model.h5 in the app root to enable segmentation inference.")

    up_img = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="spacenet_img")
    if up_img is not None:
        # Show a smaller, centered preview for a more polished look
        left, mid, right = st.columns([1, 2, 1])
        with mid:
            st.image(up_img, caption="Uploaded image", width=380)

    predict_disabled = (model is None or up_img is None)
    if st.button("Predict", use_container_width=True, disabled=predict_disabled):
        pil_img = Image.open(up_img)
        pil_mask = None  # mask upload removed
        with st.spinner('Running inference...'):
            out = _run_segmentation(model, pil_img, pil_mask)

        c1, c2, c3 = st.columns(3)
        c1.image(out['original'], caption='Original', use_column_width=True)
        c2.image(out['pred_colored'], caption='Predicted mask (colored)', use_column_width=True)
        c3.image(out['overlay'], caption='Overlay', use_column_width=True)
        # Metrics are not shown since mask upload is removed

        # Show land-use result (Residential or Commercial)
        label = classify_landuse(out['pred_mask'])
        st.success(f"Predicted land use: {label}")

        # Prepare a single PDF with all outputs
        def _to_rgb_pil(arr: np.ndarray) -> Image.Image:
            # Convert HxW or HxWxC array in [0..1] or [0..255] to RGB PIL.Image
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr8 = arr.clip(0, 255).astype(np.uint8)
            else:
                arr8 = arr
            if arr8.ndim == 2:
                im = Image.fromarray(arr8, mode='L').convert('RGB')
            else:
                im = Image.fromarray(arr8).convert('RGB')
            return im

        pdf_images = [
            _to_rgb_pil(out['original']),
            _to_rgb_pil(out['pred_colored']),
            _to_rgb_pil(out['overlay']),
        ]

        buf = io.BytesIO()
        # Save as a multi-page PDF, one image per page
        pdf_images[0].save(buf, format='PDF', resolution=300.0, save_all=True, append_images=pdf_images[1:])
        pdf_bytes = buf.getvalue()

        st.subheader("Download results")
        st.download_button(
            label='Download results PDF',
            data=pdf_bytes,
            file_name='spacenet_results.pdf',
            mime='application/pdf',
            use_container_width=True,
        )


def features_page():
    st.header("Features")
    st.caption("Explore the core capabilities of the platform.")

    # Prepare feature icons
    hp_uri = _data_uri_from_file(ASSETS_DIR / 'feature_house_price.svg')
    ts_uri = _data_uri_from_file(ASSETS_DIR / 'feature_time_series.svg')
    sn_uri = _data_uri_from_file(ASSETS_DIR / 'feature_spacenet.svg')

    cols = st.columns(3, gap="large")
    with cols[0]:
        st.markdown(f'<div class="glass-card equal-card"><div>'
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'{f"<img src=\"{hp_uri}\" alt=\"House Price\" style=\"width:42px;height:42px;\">" if hp_uri else ""}'
                    f'<h4 class="features-title" style="margin:0;">House Price Prediction</h4></div>'
                    '<div class="muted">Upload a CSV and estimate sale prices using the latest model.</div>'
                    '</div><div>', unsafe_allow_html=True)
        if st.button("Open", key="feat_hp", use_container_width=True):
            st.session_state.page = "House price prediction"
        st.markdown('</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="glass-card equal-card"><div>'
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'{f"<img src=\"{ts_uri}\" alt=\"Time Series\" style=\"width:42px;height:42px;\">" if ts_uri else ""}'
                    f'<h4 class="features-title" style="margin:0;">Time Series Forecasting</h4></div>'
                    '<div class="muted">Predict next-month ZHVI at the city level from your data.</div>'
                    '</div><div>', unsafe_allow_html=True)
        if st.button("Open", key="feat_ts", use_container_width=True):
            st.session_state.page = "Time series forecasting"
        st.markdown('</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="glass-card equal-card"><div>'
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'{f"<img src=\"{sn_uri}\" alt=\"Spacenet\" style=\"width:42px;height:42px;\">" if sn_uri else ""}'
                    f'<h4 class="features-title" style="margin:0;">Spacenet Segmentation</h4></div>'
                    '<div class="muted">Segment buildings and classify land use from satellite tiles.</div>'
                    '</div><div>', unsafe_allow_html=True)
        if st.button("Open", key="feat_sn", use_container_width=True):
            st.session_state.page = "Spacenet"
        st.markdown('</div></div>', unsafe_allow_html=True)


def about_page():
    st.header("About")
    html = """
<div class="glass-card">
<h3 style="margin-top:0;">RealtyAI</h3>
<p>RealtyAI is an intelligent real estate analytics platform that leverages Artificial Intelligence to deliver actionable insights for property buyers, investors, and urban planners. The system combines advanced computer vision and machine learning models to analyze both satellite imagery and tabular property data, offering a comprehensive understanding of real estate markets.</p>
<p>Our platform evaluates property conditions, predicts price trends, and segments satellite images to identify residential and commercial zones. With an integrated dashboard, users can explore visual analytics, forecasts, and property insights—all in one place.</p>
<h4>Key Capabilities</h4>
<ul>
<li>📈 <strong>Price Prediction:</strong> Uses regression models like XGBoost and LightGBM to estimate accurate property values.</li>
<li>🌍 <strong>Satellite Image Segmentation:</strong> Detects and labels regions using deep learning architectures such as U-Net and DeepLab.</li>
<li>⏳ <strong>Market Trend Forecasting:</strong> Predicts future price movements using LSTM and Prophet models.</li>
</ul>
<h4>Built With</h4>
<p class="muted">Python · TensorFlow/PyTorch · OpenCV · Pandas · scikit-learn · GeoPandas · Streamlit</p>
<h4>Our Mission</h4>
<p>To make property intelligence smarter, data-driven, and visually accessible—empowering users to make confident, informed real estate decisions.</p>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def contact_page():
    st.header("Contact")
    html = """
<div class="glass-card">
<h3 style="margin-top:0;">📬 Contact Me</h3>
<p>Have a question or suggestion about <strong>RealtyAI</strong>? I'd love to hear from you!</p>
<p>
📧 <a href="https://mail.google.com/mail/?view=cm&fs=1&to=varad.realtyai@gmail.com&su=RealtyAI%20Project%20Inquiry" target="_blank" rel="noopener"><strong>Open Gmail Compose</strong></a>
&nbsp;·&nbsp;

🌐 <a href="https://github.com/varad1004" target="_blank"><strong>GitHub Profile</strong></a><br>
📍 <strong>Location:</strong> Mumbai, India</p>
<p><em>RealtyAI is a project focused on using AI to deliver smart, data-driven insights for the real estate market.
Your feedback is always welcome — let’s make real estate smarter together!</em></p>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ---------- Navigation ----------

# Render the top nav bar
render_top_nav()

# Resolve page and route
page = st.session_state.get("page", "Home")

if page == "Home":
    home_page()
elif page == "Features":
    features_page()
elif page == "About":
    about_page()
elif page == "Contact":
    contact_page()
elif page == "House price prediction":
    house_price_page()
elif page == "Time series forecasting":
    time_series_page()
elif page == "Spacenet":
    spacenet_page()
else:
    home_page()

# Footer
st.markdown(
    """
    <div class="app-footer">
      RealtyAI • v1.0.0 • Crafted with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)

