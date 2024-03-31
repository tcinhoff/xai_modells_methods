from .base_model import BaseModel
from sklearn.neural_network import MLPRegressor


class MLPModel(BaseModel):
    def __init__(self, train, target_col="yhat", config=None):
        super().__init__(train, target_col)
        self.model = MLPRegressor(
            hidden_layer_sizes=[94, 71], # Durch Hyperparameteroptimierung ermittelt
            #[190, 133, 68, 75, 65] alternativ, wenn man preprocessing macht
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            max_iter=1000,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )

    def fit(self):
        self.model.fit(self.X, self.y)

    def predict(self, test):
        return self.model.predict(test)
