"""
Test the temporal text feature in load_datav3
"""

from src.dataset.dataloader import load_datav3
from src.utils.temporal_features import timestamp_to_text
import datetime

# Test the timestamp_to_text function with different times
print("=" * 70)
print("TEMPORAL TEXT EXAMPLES")
print("=" * 70)

test_timestamps = [
    1443871079,  # Saturday 11am, Fall 2015
    1609459200,  # Friday 12pm, Winter 2021 (New Year's Day)
    1625097600,  # Thursday 12pm, Summer 2021
    1617235200,  # Thursday 12am (midnight), Spring 2021
    1624744800,  # Saturday 8pm, Summer 2021
]

for ts in test_timestamps:
    dt = datetime.datetime.fromtimestamp(ts)
    temporal_text = timestamp_to_text(ts)
    print(f"\nTimestamp: {ts}")
    print(f"Actual Date: {dt.strftime('%Y-%m-%d %I:%M %p (%A)')}")
    print(f"Generated Text: {temporal_text}")

# Load a small sample with temporal features
print("\n" + "=" * 70)
print("LOADING DATA WITH TEMPORAL TEXT")
print("=" * 70)

data_dir = "data/influencer_dataset_top20_min100posts_nonsponsored"
text_dir = "checkpoints/run_20260314_124407/it_3/text_output"
df = load_datav3(data_dir, classification_task="perf", text_dir=text_dir, include_user_info=True)

if not df.empty:
    print(f"\nLoaded {len(df)} samples")
    print("\nSample text with temporal features:")
    print("-" * 70)
    sample_text = df.iloc[0]["text"]
    # Show just a snippet
    if len(sample_text) > 500:
        print(sample_text[:500] + "...")
    else:
        print(sample_text)
