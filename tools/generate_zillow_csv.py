import csv
from pathlib import Path
from datetime import datetime
import calendar

# Input and output paths
IN_PATH = Path(r"c:\\Users\\USER\\OneDrive\\Desktop\\realtyAI\\models\\test_split_ZHVI_AllHomes_20251031_213633.csv")
OUT_PATH = Path(r"c:\\Users\\USER\\OneDrive\\Desktop\\realtyAI\\zillow2.csv")

# Config: multipliers for synthetic columns
MULT_3BED = 1.23
MULT_BOTTOM = 0.89
MULT_TOP = 1.54

DECREASING_PCT = 29.84
INCREASING_PCT = 59.95


def round_to_100(x: float) -> int:
    return int(round(x / 100.0)) * 100


def _increment_month(date_str: str) -> str:
    """Increment a YYYY-MM-DD date string by one calendar month, clamping the day.
    If parsing fails, return the original string unchanged.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return date_str

    year = dt.year + (dt.month // 12)
    month = 1 if dt.month == 12 else dt.month + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day).strftime("%Y-%m-%d")


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    rows_out = []
    first_region = None

    with IN_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row.get("RegionName")
            target_date = row.get("TargetDate")
            # Use true values for base ZHVI (could also choose predicted)
            try:
                allhomes = float(row.get("y_true_ZHVI_AllHomes") or 0)
            except ValueError:
                continue

            if first_region is None:
                first_region = region

            # Only collect rows for the first encountered region; keep scanning entire file
            if region != first_region:
                continue

            # Build Zillow-style row
            zhvi_middle = round_to_100(allhomes)
            zhvi_sfr = zhvi_middle  # proxy
            zhvi_3bed = round_to_100(allhomes * MULT_3BED)
            zhvi_bottom = round_to_100(allhomes * MULT_BOTTOM)
            zhvi_top = round_to_100(allhomes * MULT_TOP)

            rows_out.append({
                "PctOfHomesDecreasingInValues_AllHomes": f"{DECREASING_PCT:.2f}",
                "PctOfHomesIncreasingInValues_AllHomes": f"{INCREASING_PCT:.2f}",
                "ZHVI_3bedroom": zhvi_3bed,
                "ZHVI_BottomTier": zhvi_bottom,
                "ZHVI_MiddleTier": zhvi_middle,
                "ZHVI_SingleFamilyResidence": zhvi_sfr,
                "ZHVI_TopTier": zhvi_top,
                "Date": target_date,
            })

    if not rows_out:
        raise RuntimeError("No rows produced. Check input format and columns.")

    # Sort by date (ascending) if parseable
    def _date_key(r: dict):
        d = r.get("Date") or ""
        try:
            return datetime.strptime(d, "%Y-%m-%d")
        except Exception:
            return datetime.min

    rows_out.sort(key=_date_key)

    # Ensure exactly 101 rows: truncate or pad by repeating last row with next-month date
    DESIRED = 101
    if len(rows_out) > DESIRED:
        rows_out = rows_out[:DESIRED]
    elif len(rows_out) < DESIRED and rows_out:
        last_date = rows_out[-1].get("Date")
        last_row = rows_out[-1]
        for _ in range(DESIRED - len(rows_out)):
            new_row = dict(last_row)
            new_date = _increment_month(last_date) if last_date else last_date
            new_row["Date"] = new_date
            rows_out.append(new_row)
            last_date = new_date

    # Write output CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "PctOfHomesDecreasingInValues_AllHomes",
        "PctOfHomesIncreasingInValues_AllHomes",
        "ZHVI_3bedroom",
        "ZHVI_BottomTier",
        "ZHVI_MiddleTier",
        "ZHVI_SingleFamilyResidence",
        "ZHVI_TopTier",
        "Date",
    ]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
