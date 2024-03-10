from .base_model import BaseModel
from lightgbm import LGBMRegressor


class LGBMModel(BaseModel):
    LGBM_PARAMS = {
        "num_leaves": {"default": 31, "min": 2, "max": 131070, "step": 1},
        "learning_rate": {
            "default": 0.01,
            "min": 0.00001,
            "max": 100000,
            "step": 0.00001,
        },
        "n_estimators": {"default": 100, "min": 1, "max": 100000, "step": 1},
        "max_depth": {"default": -1, "min": -1, "max": 100000, "step": 1},
        "min_child_samples": {"default": 20, "min": 1, "max": 100000, "step": 1},
        "reg_alpha": {"default": 0.0, "min": 0.0, "max": 100000, "step": 0.00001},
        "reg_lambda": {"default": 0.0, "min": 0.0, "max": 100000, "step": 0.00001},
        "subsample": {"default": 1.0, "min": 0.00001, "max": 1.0, "step": 0.00001},
    }

    def __init__(self, train, target_col="yhat", config=None):
        super().__init__(train, target_col)
        self.model = (
            LGBMRegressor(verbose=-1, importance_type="gain")
            if config is None
            else LGBMRegressor(**config)
        )

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, test):
        return self.model.predict(test)
