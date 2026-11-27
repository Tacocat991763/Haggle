# Haggle

A PyTorch neural network that predicts the number of hours spent developing a software project using repository statistics, programming languages, frameworks, and other project metadata.

The model uses numerical normalization, categorical encoding, dropout, regularization, and validation RMSLE-based checkpoint selection.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Place `test.csv` alongside `predict.py` and run:

```bash
python predict.py
```

The script generates `predictions.csv` containing project IDs and predicted hours spent.

Inference runs entirely on the CPU and does not require an internet connection.

Made with <3 by @Screaminbanana