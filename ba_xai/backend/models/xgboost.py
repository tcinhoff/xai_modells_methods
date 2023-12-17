import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, train, target_col="yhat"):
        super().__init__(train, target_col)
        self.model = xgb.XGBRegressor()

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, test):
        return self.model.predict(test)
