"""
Extract temporal features from Instagram post timestamps.
"""

import datetime
import logging

import numpy as np

logger = logging.getLogger(__name__)


def timestamp_to_text(timestamp: int) -> str:
    """
    Convert Unix timestamp to natural language temporal description.

    Args:
        timestamp: Unix timestamp (seconds since 1970)

    Returns:
        Natural language string like "Posted on Saturday (weekend) at 7pm in the evening during summer."
    """
    try:
        dt = datetime.datetime.fromtimestamp(timestamp)

        # Day of week
        day_name = dt.strftime("%A")  # Monday, Tuesday, etc.

        # Time of day description
        hour = dt.hour
        if 5 <= hour < 12:
            time_desc = "morning"
            time_detail = f"{hour}am"
        elif 12 <= hour < 17:
            time_desc = "afternoon"
            time_detail = f"{hour-12 if hour > 12 else 12}pm"
        elif 17 <= hour < 21:
            time_desc = "evening"
            time_detail = f"{hour-12}pm"
        else:
            time_desc = "night"
            time_detail = (
                f"{hour-12 if hour > 12 else hour}{'pm' if hour >= 12 else 'am'}"
            )

        # Season
        month = dt.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"

        # Weekend/weekday
        is_weekend = dt.weekday() >= 5
        day_type = "weekend" if is_weekend else "weekday"

        # Construct natural language description
        text = f"Posted on {day_name} ({day_type}) at {time_detail} in the {time_desc} during {season}."

        return text
    except (ValueError, OSError, OverflowError) as e:
        logger.warning("Error converting timestamp %s: %s", timestamp, e)
        return "Posted at an unknown time."


def extract_temporal_features(timestamp: int) -> dict:
    """
    Extract temporal features from Unix timestamp.

    Args:
        timestamp: Unix timestamp (seconds since 1970-01-01)

    Returns:
        Dictionary with temporal features
    """
    dt = datetime.datetime.fromtimestamp(timestamp)

    features = {
        # Time of day (0-23)
        "hour_of_day": dt.hour,
        # Day of week (0=Monday, 6=Sunday)
        "day_of_week": dt.weekday(),
        # Is weekend (Saturday=5, Sunday=6)
        "is_weekend": int(dt.weekday() >= 5),
        # Month (1-12)
        "month": dt.month,
        # Quarter (1-4)
        "quarter": (dt.month - 1) // 3 + 1,
        # Is holiday season (Nov-Dec)
        "is_holiday_season": int(dt.month in [11, 12]),
        # Is summer (Jun-Aug in Northern hemisphere)
        "is_summer": int(dt.month in [6, 7, 8]),
        # Year (for trend analysis)
        "year": dt.year,
        # Time slot (morning, afternoon, evening, night)
        "time_slot": get_time_slot(dt.hour),
        # Peak posting hours (6pm-9pm typically get most engagement)
        "is_peak_hours": int(18 <= dt.hour <= 21),
        # Business hours (9am-5pm)
        "is_business_hours": int(9 <= dt.hour <= 17),
    }

    # Cyclical encoding for periodic features (better for ML)
    # Hour of day (24-hour cycle)
    features["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)

    # Day of week (7-day cycle)
    features["day_sin"] = np.sin(2 * np.pi * dt.weekday() / 7)
    features["day_cos"] = np.cos(2 * np.pi * dt.weekday() / 7)

    # Month (12-month cycle)
    features["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * dt.month / 12)

    return features


def get_time_slot(hour: int) -> int:
    """
    Convert hour to time slot:
    0 = night (0-5)
    1 = morning (6-11)
    2 = afternoon (12-17)
    3 = evening (18-23)
    """
    if 0 <= hour < 6:
        return 0  # night
    elif 6 <= hour < 12:
        return 1  # morning
    elif 12 <= hour < 18:
        return 2  # afternoon
    else:
        return 3  # evening


def add_temporal_features_to_df(df, timestamp_col="taken_at_timestamp"):
    """
    Add temporal features to a pandas DataFrame.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added temporal features
    """
    import pandas as pd

    # Extract all temporal features
    temporal_data = df[timestamp_col].apply(extract_temporal_features)

    # Convert to DataFrame
    temporal_df = pd.DataFrame(temporal_data.tolist())

    # Concatenate with original DataFrame
    df_with_temporal = pd.concat([df, temporal_df], axis=1)

    return df_with_temporal


if __name__ == "__main__":
    # Example usage
    timestamp = 1443871079  # Oct 3, 2015, 14:04:39

    features = extract_temporal_features(timestamp)

    print("Temporal features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    # Show what day/time it was
    dt = datetime.datetime.fromtimestamp(timestamp)
    print(f"\nOriginal timestamp: {timestamp}")
    print(f"Date/Time: {dt}")
    print(f"Day: {dt.strftime('%A')}")
    print(f"Time: {dt.strftime('%I:%M %p')}")
