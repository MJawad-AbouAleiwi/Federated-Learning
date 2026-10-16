# The federated server, samples clients and aggregates with FedAvg
import numpy as np

from src.model import SoftmaxRegression

def fed_avg(updates):
    # Weighted average of client weights
    total = sum(n for _, n in updates)
    n_tensors = len(updates[0][0])
    return [
        sum(w[i] * (n / total) for w, n in updates)
        for i in range(n_tensors)
    ]

class Server:
    def __init__(self, clients, n_features, n_classes, seed=0):
        self.clients = clients
        self.global_model = SoftmaxRegression(n_features, n_classes, seed=seed)
        self.rng = np.random.default_rng(seed)

    def run_round(self, round_idx, clients_per_round, epochs, lr, batch_size):
        chosen = self.rng.choice(len(self.clients), size=clients_per_round,
                                 replace=False)
        global_weights = self.global_model.get_weights()
        updates = [
            self.clients[i].train(global_weights, epochs, lr, batch_size,
                                  seed=round_idx * 1000 + int(i))
            for i in chosen
        ]
        self.global_model.set_weights(fed_avg(updates))