from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from .base_model import BaseModel

class SklearnGAM(BaseModel):
    def __init__(self, train, target_col="yhat", config=None):
        super().__init__(train, target_col)
        # Pipeline, die Spline-Transformation mit einem linearen Modell kombiniert
        self.model = make_pipeline(SplineTransformer(), LinearRegression())

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, test):
        return self.model.predict(test)
