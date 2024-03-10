import xgboost as xgb
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    XGBOOST_PARAMS = {
        "n_estimators": {"default": 100, "min": 1, "max": 100000, "step": 1},
        "learning_rate": {
            "default": 0.01,
            "min": 0.00001,
            "max": 100000,
            "step": 0.00001,
        },
        "max_depth": {"default": 6, "min": 0, "max": 100000, "step": 1},
        "min_child_weight": {"default": 0, "min": 0, "max": 100000, "step": 1},
        "gamma": {"default": 0, "min": 0, "max": 100000, "step": 0.000001},
        "subsample": {"default": 1, "min": 0.0, "max": 1, "step": 0.000001},
        "colsample_bytree": {"default": 1, "min": 0.0, "max": 1, "step": 0.000001},
    }

    def __init__(self, train, target_col="yhat", config=None):
        super().__init__(train, target_col)
        self.model = (
            xgb.XGBRegressor() if config is None else xgb.XGBRegressor(**config)
        )

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, test):
        return self.model.predict(test)
