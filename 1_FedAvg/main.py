# Federated Averaging demo on digits dataset
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_data, partition_iid, partition_non_iid
from src.client import Client
from src.server import Server
from src.model import SoftmaxRegression

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--clients-per-round", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--iid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.seed)
    n_features, n_classes = X_train.shape[1], 10

    parts = (partition_iid(X_train, y_train, args.clients, args.seed) if args.iid
             else partition_non_iid(X_train, y_train, args.clients, 2, args.seed))
    clients = [Client(i, X, y, n_features, n_classes)
               for i, (X, y) in enumerate(parts)]

    print(f"{'IID' if args.iid else 'Non-IID'} split | "
          f"{args.clients} clients | samples per client: "
          f"{[c.n_samples for c in clients]}")

    server = Server(clients, n_features, n_classes, seed=args.seed)
    history = []
    for r in range(1, args.rounds + 1):
        server.run_round(r, args.clients_per_round, args.local_epochs,
                         args.lr, args.batch_size)
        loss, acc = server.global_model.evaluate(X_test, y_test)
        history.append((loss, acc))
        if r % 5 == 0 or r == 1:
            print(f"Round {r:3d} | test loss {loss:.4f} | test acc {acc:.3f}")

    # Baseline, centralized training on all pooled data
    central = SoftmaxRegression(n_features, n_classes)
    central.fit(X_train, y_train, epochs=args.rounds * args.local_epochs,
                lr=args.lr, batch_size=args.batch_size)
    _, central_acc = central.evaluate(X_test, y_test)
    print(f"\nFederated final accuracy:   {history[-1][1]:.3f}")
    print(f"Centralized baseline:       {central_acc:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot([h[1] for h in history], label="Federated (FedAvg)")
    plt.axhline(central_acc, color="gray", ls="--", label="Centralized")
    plt.xlabel("Communication round"); plt.ylabel("Test accuracy")
    plt.title("FedAvg on digits (" + ("IID" if args.iid else "non-IID") + ")")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/accuracy.png", dpi=120)
    print("Saved plot to results/accuracy.png")

if __name__ == "__main__":
    main()