class BaseModel:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.X = train.drop(columns=["yhat"])
        self.y = train["yhat"]

    def fit(self):
        pass

    def predict(self):
        pass
