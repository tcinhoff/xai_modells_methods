from .base_model import BaseModel
from lightgbm import LGBMRegressor


class LGBMModel(BaseModel):
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
