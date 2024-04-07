from .base_model import BaseModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, Sum
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from app_instance import PATHS


class GPRModel(BaseModel):
    def __init__(self, train, target_col="yhat", kernel=None, config=None, alpha=0.1):
        super().__init__(train, target_col)
        if kernel is None:
            # Kernel-Kombination aus periodischem und Matern-Kernel
            # daily_kernel = ExpSineSquared(length_scale=1.0, periodicity=10)
            # weekly_kernel = ExpSineSquared(length_scale=7.0, periodicity=70)
            matern_kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)
            
            # periodic_kernel_sum = Sum(daily_kernel, weekly_kernel)
            # kernel = Sum(daily_kernel, matern_kernel)
        self.model = GaussianProcessRegressor(kernel=matern_kernel)
        self.scaler = StandardScaler()

    def fit(self):
        # Normalisieren der Trainingsdaten
        X = self.scaler.fit_transform(self.X)

        with open(PATHS["path_to_scaler"], "wb") as f:
            pickle.dump(self.scaler, f)

        X = pd.DataFrame(X, columns=self.X.columns)

        self.model.fit(X, self.y)

    def predict(self, test, return_std=False):
        test_normalized = pd.DataFrame(
            self.scaler.transform(test), columns=test.columns
        )
        if return_std:
            predictions, std_dev = self.model.predict(test_normalized, return_std=return_std)
            return predictions, std_dev
        else:
            return self.model.predict(test_normalized)
