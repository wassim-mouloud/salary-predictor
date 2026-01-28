"""
Linear Regression model for Salary Prediction.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SalaryPredictor:
    """Linear Regression model for predicting salaries."""

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_columns = ['years_experience', 'education_level', 'age', 'job_role']
        self.is_fitted = False

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare features from dataframe."""
        return df[self.feature_columns].values

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Train the model and return evaluation metrics.

        Returns dict with train/test metrics.
        """
        X = self.prepare_features(df)
        y = df['salary'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred)
            },
            'test': {
                'r2': r2_score(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred)
            },
            'coefficients': dict(zip(self.feature_columns, self.model.coef_)),
            'intercept': self.model.intercept_,
            'n_train': len(y_train),
            'n_test': len(y_test)
        }

        return metrics

    def predict(self, years_experience: float, education_level: int,
                age: int, job_role: int) -> float:
        """Predict salary for given features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array([[years_experience, education_level, age, job_role]])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict salaries for multiple samples."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: str):
        """Save the model and scaler to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, path)

    def load(self, path: str):
        """Load the model and scaler from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = True


def train_and_save_model():
    """Train the model and save it."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "salary_data.csv"
    model_path = project_root / "models" / "salary_model.joblib"

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")

    predictor = SalaryPredictor()
    metrics = predictor.fit(df)

    print("\n" + "=" * 50)
    print("MODEL TRAINING RESULTS")
    print("=" * 50)
    print(f"\nTraining samples: {metrics['n_train']}")
    print(f"Test samples: {metrics['n_test']}")

    print("\n--- Training Metrics ---")
    print(f"R² Score: {metrics['train']['r2']:.4f}")
    print(f"RMSE: ${metrics['train']['rmse']:,.2f}")
    print(f"MAE: ${metrics['train']['mae']:,.2f}")

    print("\n--- Test Metrics ---")
    print(f"R² Score: {metrics['test']['r2']:.4f}")
    print(f"RMSE: ${metrics['test']['rmse']:,.2f}")
    print(f"MAE: ${metrics['test']['mae']:,.2f}")

    print("\n--- Feature Coefficients ---")
    for feature, coef in metrics['coefficients'].items():
        print(f"{feature}: {coef:,.2f}")
    print(f"Intercept: {metrics['intercept']:,.2f}")

    model_path.parent.mkdir(exist_ok=True)
    predictor.save(model_path)
    print(f"\nModel saved to {model_path}")

    return predictor, metrics


if __name__ == "__main__":
    train_and_save_model()
