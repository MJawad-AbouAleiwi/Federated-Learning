# Federated Averaging - FedAvg

An implementation of **Federated Averaging (FedAvg)**, a federated learning
algorithm in which many clients train a shared model on their own private data
and a central server averages their model weights, so raw data never leaves
the clients.

## How it Works
1. The server holds a global model.
2. Each round, it picks a subset of clients and sends them the global weights.
3. Each client trains locally on its own private data.
4. Clients send back updated weights, the server averages them, weighted by dataset size.

## Project Structure
```
Federated Averaging/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── client.py
│   └── server.py
└── results/
    └── accuracy.png
```

## Run
```bash
pip install -r requirements.txt
python main.py            # non-IID, Each client sees ~2 digits
python main.py --iid      # IID
```
