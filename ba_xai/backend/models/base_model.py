class BaseModel:
    def __init__(self, train, target_col="yhat"):
        self.train = train
        self.X = train.drop(columns=[target_col])
        self.y = train[target_col]

    def fit(self):
        pass

    def predict(self, test):
        pass
