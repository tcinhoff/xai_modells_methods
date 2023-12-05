from .base_model import BaseModel
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(BaseModel):
    def __init__(self, train, test):
        super().__init__(train, test)
        self.model = LinearRegression()

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self):
        return self.model.predict(self.test)
