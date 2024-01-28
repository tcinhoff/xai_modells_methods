from .base_model import BaseModel
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

class TabNetModel(BaseModel):
    def __init__(self, train, target_col="yhat", config=None):
        super().__init__(train, target_col)

        default_config = {
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=2e-2),
            "mask_type": 'entmax'  # "sparsemax"
        }
        if config is None:
            config = default_config

        self.model = TabNetRegressor(**config)

    def fit(self):
        X_train = self.X.values
        y_train = self.y.values.reshape(-1, 1)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            eval_name=['train'],
            eval_metric=['rmse']
        )

    def predict(self, test):
        X_test = test.values
        predictions = self.model.predict(X_test)
        return predictions.ravel()
