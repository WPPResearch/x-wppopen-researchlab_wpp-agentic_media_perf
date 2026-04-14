import os
import matplotlib.pyplot as plt
from src.dataset.dataloader import convert_to_classification_labels


# Helper to save plot
def save_plot(fig, name):
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path)
    print(f"Saved plot to {path}")
    plt.close(fig)


def plot_engagement_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Check if y is continuous or categorical
    # If called before conversion, y is continuous.
    ax.hist(df["y"], bins=30, color="skyblue", edgecolor="black")
    ax.set_title("Distribution of Engagement Rates")
    ax.set_xlabel("Engagement Rate")
    ax.set_ylabel("Frequency")
    save_plot(fig, "engagement_distribution.png")


def plot_class_distribution(df):
    if "y" not in df.columns:
        return

    # Check if conversion happened (y should be 0, 1, 2)
    # We can check unique values or if y_original exists
    if "y_original" not in df.columns:
        print(
            "Warning: Dataframe does not seem to have classification labels (y_original missing). Skipping class plot."
        )
        return

    counts = df["y"].value_counts().sort_index()
    # Ensure all classes are present for consistent coloring if possible,
    # but value_counts only shows present ones.

    labels_map = {0: "Low (Bottom 15%)", 1: "Average (Middle 70%)", 2: "High (Top 15%)"}
    labels = [labels_map.get(i, str(i)) for i in counts.index]

    colors = [
        "salmon" if i == 0 else "gold" if i == 1 else "lightgreen" for i in counts.index
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    counts.plot(kind="bar", color=colors, ax=ax, edgecolor="black")
    ax.set_title("Class Distribution (Success Tiers)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(labels, rotation=0)

    # Add counts on top
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    save_plot(fig, "class_distribution.png")


def plot_text_length_vs_engagement(df):
    # Calculate text length
    df["text_length"] = df["text"].apply(len)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Use y_original if available (continuous)
    y_col = "y_original" if "y_original" in df.columns else "y"

    ax.scatter(df["text_length"], df[y_col], alpha=0.5, c="purple", s=10)
    ax.set_title("Text Length vs Engagement Rate")
    ax.set_xlabel("Text Length (chars)")
    ax.set_ylabel("Engagement Rate")

    # Optional: Log scale if skewed
    # ax.set_yscale('log')

    save_plot(fig, "text_length_vs_engagement.png")


if __name__ == "__main__":
    from src.utils.cli import create_data_argparser, load_data_from_args

    parser = create_data_argparser("Generate data analytics diagrams.")
    args = parser.parse_args()

    df = load_data_from_args(args)

    if df is not None and not df.empty:
        print("Generating Engagement Distribution Plot...")
        plot_engagement_distribution(df)

        print("Converting to classes...")
        df = convert_to_classification_labels(df)

        print("Generating Class Distribution Plot...")
        plot_class_distribution(df)

        print("Generating Text Length vs Engagement Plot...")
        plot_text_length_vs_engagement(df)

        print("\nAll plots generated in 'plots/' directory.")
    else:
        print("Failed to load data or empty dataframe.")
