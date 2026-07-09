from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import numpy as np
from src.utils.experiment_tracker import ExperimentTracker, create_experiment_summary
from src.utils.embeddings import generate_embeddings
from src.utils.metrics import log_sample_predictions
from src.training.model_configs import DEFAULT_XGBOOST_PARAMS, DEFAULT_LIGHTGBM_PARAMS


def train_and_evaluate_classifiers(df):
    if df.empty:
        print("No data loaded.")
        return

    print(f"Loaded {len(df)} samples for classification.")

    y = df["y"]
    X_train = generate_embeddings(df["text"].tolist())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    best_model_name = ""
    best_accuracy = 0.0

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        pipeline = Pipeline([("classifier", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    print(f"\nBest Classifier: {best_model_name} with Accuracy: {best_accuracy:.4f}")


def train_and_evaluate_models(df, embedding_model="all-mpnet-base-v2", dataset_name=""):
    if df.empty:
        print("No data loaded.")
        return

    # Create experiment tracker
    with ExperimentTracker("text_embedding_regression") as tracker:
        tracker.log(f"Loaded {len(df)} samples.")
        y = np.log(df["y"] + 1.0)
        tracker.log(f"Target stats:\n{df['y'].describe()}")
        tracker.log(f"Log target distribution:\n{np.log(df['y'] + 1.0).describe()}")
        tracker.log(f"Variance: {np.log(df['y'] + 1.0).var()}")

        # Save parameters
        params = {
            "n_samples": len(df),
            "embedding_model": embedding_model,
            "test_size": 0.2,
            "random_state": 42,
            "target_transform": "log(likes + 1)",
        }
        tracker.save_params(params)

        X_train = generate_embeddings(df["text"].tolist(), model_name=embedding_model)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y, test_size=0.2, random_state=42
        )

        tracker.log(f"Training set size: {len(X_train)}")
        tracker.log(f"Test set size: {len(X_test)}")

        # Try different Ridge alphas (regularization strength)
        models = {
            "Ridge (alpha=0.1)": Ridge(alpha=0.1),
            "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        }

        # Try XGBoost if available (often better than GradientBoosting)
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(**DEFAULT_XGBOOST_PARAMS)
            print("XGBoost available - added to models")
        except ImportError:
            print(
                "XGBoost not installed - skipping (install with: pip install xgboost)"
            )

        # Try LightGBM if available (fast and often best for tabular data)
        try:
            from lightgbm import LGBMRegressor

            models["LightGBM"] = LGBMRegressor(**DEFAULT_LIGHTGBM_PARAMS)
            print("LightGBM available - added to models")
        except ImportError:
            print(
                "LightGBM not installed - skipping (install with: pip install lightgbm)"
            )

        best_model_name = ""
        best_r2 = -float("inf")
        best_model = None
        best_predictions = None
        all_model_metrics = {}

        for name, model in models.items():
            tracker.log(f"\n--- Training {name} ---")

            # Build pipeline
            pipeline = Pipeline([("regressor", model)])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store metrics
            model_metrics = {
                f"{name}_MAE": mae,
                f"{name}_RMSE": rmse,
                f"{name}_R2": r2,
            }
            all_model_metrics.update(model_metrics)

            tracker.log(f"MAE: {mae:.2f}")
            tracker.log(f"RMSE: {rmse:.2f}")
            tracker.log(f"R2 Score: {r2:.4f}")

            # Show sample predictions
            log_sample_predictions(y_test, y_pred, n=10, log_fn=tracker.log)

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model = pipeline
                best_predictions = y_pred

        # Save all metrics
        tracker.save_metrics(all_model_metrics, stage="test")

        # Save best model
        if best_model is not None:
            tracker.save_model(
                best_model, model_name=f"best_{best_model_name.replace(' ', '_')}"
            )
            tracker.save_predictions(y_test.values, best_predictions, stage="test")

        # Create and save summary
        summary = create_experiment_summary(
            "Text Embedding Regression",
            params,
            {"Best_Model": best_model_name, "Best_R2": best_r2, **all_model_metrics},
            best_model_name,
        )
        tracker.save_summary(summary)

        tracker.log(f"\nBest Model: {best_model_name} with R2: {best_r2:.4f}")


if __name__ == "__main__":
    from src.utils.cli import create_data_argparser, load_data_from_args

    parser = create_data_argparser("Train engagement prediction models.")
    args = parser.parse_args()

    df = load_data_from_args(args)
    print(df.head())
    print("regression")
    train_and_evaluate_models(df, embedding_model=args.embedding)
