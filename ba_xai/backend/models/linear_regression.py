from .base_model import BaseModel
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(BaseModel):
    def __init__(self, train, target_col="yhat"):
        super().__init__(train, target_col)
        self.model = LinearRegression()

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, test):
        return self.model.predict(test)
