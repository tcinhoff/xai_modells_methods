from .base_model import BaseModel
from lightgbm import LGBMRegressor


class LGBMModel(BaseModel):
    def __init__(self, train, test):
        super().__init__(train, test)
        self.model = LGBMRegressor()

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self):
        return self.model.predict(self.test)
