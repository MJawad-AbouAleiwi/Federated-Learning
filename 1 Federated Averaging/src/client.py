# A federated client, trains locally, never shares raw data
from src.model import SoftmaxRegression

class Client:
    def __init__(self, client_id, X, y, n_features, n_classes):
        self.id = client_id
        self.X, self.y = X, y
        self.model = SoftmaxRegression(n_features, n_classes)

    @property
    def n_samples(self):
        return len(self.X)

    def train(self, global_weights, epochs, lr, batch_size, seed):
        # Receive global weights, train locally, return updated weights
        self.model.set_weights(global_weights)
        self.model.fit(self.X, self.y, epochs=epochs, lr=lr,
                       batch_size=batch_size, seed=seed)
        return self.model.get_weights(), self.n_samples