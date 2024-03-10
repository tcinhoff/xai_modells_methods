from .base_model import BaseModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, Sum


class GPRModel(BaseModel):
    def __init__(self, train, target_col="yhat", kernel=None, config=None, alpha=0.1):
        super().__init__(train, target_col)
        if kernel is None:
            # Kernel-Kombination aus periodischem und Matern-Kernel
            periodic_kernel = ExpSineSquared(length_scale=50.0, periodicity=10, length_scale_bounds=(0.1, 100.0))
            periodic_kernel = ExpSineSquared(length_scale=7.0, periodicity=70, length_scale_bounds=(0.1, 10.0))
            matern_kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)
            periodic_kernel_sum = Sum(periodic_kernel, matern_kernel)
            kernel = Sum(periodic_kernel_sum, matern_kernel)
        self.model = GaussianProcessRegressor(kernel=matern_kernel, normalize_y=True)

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, test, return_std=True):
        X_test = test.drop(
            columns=[col for col in [self.y.name] if col in test.columns]
        )
        if return_std:
            predictions, std_dev = self.model.predict(X_test, return_std=True)
            return predictions, std_dev
        else:
            return self.model.predict(X_test)
