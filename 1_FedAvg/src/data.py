# Load digits dataset and split it across simulated clients
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(seed=0):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), y_train, scaler.transform(X_test), y_test

def partition_iid(X, y, n_clients, seed=0):
    # Randomly shuffle and split data evenly, every client sees all classes
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return [(X[s], y[s]) for s in np.array_split(idx, n_clients)]

def partition_non_iid(X, y, n_clients, classes_per_client=2, seed=0):
    # Each client only sees a few digit classes
    rng = np.random.default_rng(seed)
    order = np.argsort(y, kind="stable")
    n_shards = n_clients * classes_per_client
    shards = np.array_split(order, n_shards)
    shard_ids = rng.permutation(n_shards)
    clients = []
    for c in range(n_clients):
        ids = shard_ids[c * classes_per_client:(c + 1) * classes_per_client]
        idx = np.concatenate([shards[i] for i in ids])
        clients.append((X[idx], y[idx]))
    return clients