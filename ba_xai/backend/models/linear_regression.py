from .base_model import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from app_instance import PATHS


class LinearRegressionModel(BaseModel):
    def __init__(self, train, target_col="yhat"):
        super().__init__(train, target_col)
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def fit(self):
        # Normalisieren der Trainingsdaten
        X = self.scaler.fit_transform(self.X)

        with open(PATHS["path_to_scaler"], "wb") as f:
            pickle.dump(self.scaler, f)

        X = pd.DataFrame(X, columns=self.X.columns)
        self.model.fit(X, self.y)

    def predict(self, test):
        test_normalized = pd.DataFrame(
            self.scaler.transform(test), columns=test.columns
        )
        return self.model.predict(test_normalized)
