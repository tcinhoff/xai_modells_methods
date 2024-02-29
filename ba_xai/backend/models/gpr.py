from .base_model import BaseModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPRModel(BaseModel):
    def __init__(self, train, target_col="yhat", kernel=None, config=None):
        super().__init__(train, target_col)
        if kernel is None:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.model = GaussianProcessRegressor(
            kernel=kernel, **(config if config else {})
        )

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
