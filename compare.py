import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)

from core.config import (
    device,
    batch_size,
    epochs,
    lr,
    seq_len,
    d_model,
    num_heads,
    num_layers,
    vocab_size
)

from core.model import LlamaModel

from core.configb import model_save_path_b
from core.modelb import BERTModel

from core.data import generate_tick_data, TickDataset


SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def build_dataset():

    X, y = generate_tick_data()

    split = int(len(X) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    train_dataset = TickDataset(
        X_train,
        y_train
    )

    test_dataset = TickDataset(
        X_test,
        y_test
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Total samples : {len(X)}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print(f"Positive rate : {y.float().mean().item():.4f}")

    return train_loader, test_loader


def train_model(model, train_loader, model_name):

    print("\n" + "=" * 60)
    print(f"TRAINING {model_name}")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2
    )

    criterion = nn.BCEWithLogitsLoss()

    losses = []
    epoch_times = []

    total_start = time.perf_counter()

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        epoch_start = time.perf_counter()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        avg_loss = (
            total_loss
            / len(train_loader)
        )

        losses.append(avg_loss)
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    return {
        "losses": losses,
        "epoch_times": epoch_times,
        "training_time": total_time
    }


def evaluate_model(model, test_loader, model_name):

    model.eval()

    all_probs = []
    all_labels = []

    X_warmup, _ = next(iter(test_loader))
    X_warmup = X_warmup.to(device)

    with torch.no_grad():

        for _ in range(5):
            model(X_warmup)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probs = torch.sigmoid(logits)

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                y_batch.numpy()
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = (
        time.perf_counter()
        - start
    )

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    predictions = (
        probs >= 0.5
    ).astype(int)

    auroc = roc_auc_score(
        labels,
        probs
    )

    pr_auc = average_precision_score(
        labels,
        probs
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    latency_ms = (
        total_time
        / len(test_loader.dataset)
        * 1000
    )

    throughput = (
        len(test_loader.dataset)
        / total_time
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    cm = confusion_matrix(
        labels,
        predictions
    )

    fpr, tpr, _ = roc_curve(
        labels,
        probs
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            labels,
            probs
        )
    )

    print("\n" + "-" * 60)
    print(f"{model_name} RESULTS")
    print("-" * 60)
    print(f"AUROC        : {auroc:.4f}")
    print(f"PR-AUC       : {pr_auc:.4f}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1           : {f1:.4f}")
    print(f"Latency      : {latency_ms:.4f} ms")
    print(f"Throughput   : {throughput:.2f} seq/s")
    print(f"Parameters   : {parameters:,}")
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "model": model_name,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": latency_ms,
        "throughput": throughput,
        "parameters": parameters,
        "confusion_matrix": cm.tolist(),
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve
    }


def plot_loss(llama_train, bert_train):

    plt.figure(figsize=(8, 5))

    x = range(1, epochs + 1)

    plt.plot(
        x,
        llama_train["losses"],
        marker="o",
        label="Llama"
    )

    plt.plot(
        x,
        bert_train["losses"],
        marker="o",
        label="BERT"
    )

    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "loss_comparison.png"
        ),
        dpi=200
    )

    plt.close()


def plot_roc(llama, bert):

    plt.figure(figsize=(7, 6))

    plt.plot(
        llama["fpr"],
        llama["tpr"],
        label=f"Llama (AUROC={llama['auroc']:.4f})"
    )

    plt.plot(
        bert["fpr"],
        bert["tpr"],
        label=f"BERT (AUROC={bert['auroc']:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "roc_comparison.png"
        ),
        dpi=200
    )

    plt.close()


def plot_pr(llama, bert):

    plt.figure(figsize=(7, 6))

    plt.plot(
        llama["recall_curve"],
        llama["precision_curve"],
        label=f"Llama (PR-AUC={llama['pr_auc']:.4f})"
    )

    plt.plot(
        bert["recall_curve"],
        bert["precision_curve"],
        label=f"BERT (PR-AUC={bert['pr_auc']:.4f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "pr_comparison.png"
        ),
        dpi=200
    )

    plt.close()


def plot_metrics(llama, bert):

    metrics = [
        "AUROC",
        "PR-AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    llama_values = [
        llama["auroc"],
        llama["pr_auc"],
        llama["accuracy"],
        llama["precision"],
        llama["recall"],
        llama["f1"]
    ]

    bert_values = [
        bert["auroc"],
        bert["pr_auc"],
        bert["accuracy"],
        bert["precision"],
        bert["recall"],
        bert["f1"]
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        llama_values,
        width,
        label="Llama"
    )

    plt.bar(
        x + width / 2,
        bert_values,
        width,
        label="BERT"
    )

    plt.xticks(
        x,
        metrics
    )

    plt.ylim(
        0,
        1.05
    )

    plt.ylabel("Score")
    plt.title("Classification Metric Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "metrics_comparison.png"
        ),
        dpi=200
    )

    plt.close()


def plot_latency(llama, bert):

    models = [
        "Llama",
        "BERT"
    ]

    latency = [
        llama["latency_ms"],
        bert["latency_ms"]
    ]

    plt.figure(figsize=(6, 5))

    plt.bar(
        models,
        latency
    )

    plt.ylabel(
        "Mean Latency (ms / sequence)"
    )

    plt.title(
        "Inference Latency Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "latency_comparison.png"
        ),
        dpi=200
    )

    plt.close()


def save_results(
    llama,
    bert,
    llama_train,
    bert_train
):

    table = pd.DataFrame({

        "Metric": [
            "AUROC",
            "PR-AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Training Time (s)",
            "Mean Latency (ms)",
            "Throughput (seq/s)",
            "Parameters"
        ],

        "Llama": [
            llama["auroc"],
            llama["pr_auc"],
            llama["accuracy"],
            llama["precision"],
            llama["recall"],
            llama["f1"],
            llama_train["training_time"],
            llama["latency_ms"],
            llama["throughput"],
            llama["parameters"]
        ],

        "BERT": [
            bert["auroc"],
            bert["pr_auc"],
            bert["accuracy"],
            bert["precision"],
            bert["recall"],
            bert["f1"],
            bert_train["training_time"],
            bert["latency_ms"],
            bert["throughput"],
            bert["parameters"]
        ]
    })

    table.to_csv(
        os.path.join(
            RESULTS_DIR,
            "comparison.csv"
        ),
        index=False
    )

    json_results = {
        "Llama": {
            "AUROC": llama["auroc"],
            "PR-AUC": llama["pr_auc"],
            "Accuracy": llama["accuracy"],
            "Precision": llama["precision"],
            "Recall": llama["recall"],
            "F1": llama["f1"],
            "Training Time": llama_train["training_time"],
            "Mean Latency": llama["latency_ms"],
            "Throughput": llama["throughput"],
            "Parameters": llama["parameters"],
            "Confusion Matrix": llama["confusion_matrix"]
        },

        "BERT": {
            "AUROC": bert["auroc"],
            "PR-AUC": bert["pr_auc"],
            "Accuracy": bert["accuracy"],
            "Precision": bert["precision"],
            "Recall": bert["recall"],
            "F1": bert["f1"],
            "Training Time": bert_train["training_time"],
            "Mean Latency": bert["latency_ms"],
            "Throughput": bert["throughput"],
            "Parameters": bert["parameters"],
            "Confusion Matrix": bert["confusion_matrix"]
        }
    }

    with open(
        os.path.join(
            RESULTS_DIR,
            "comparison.json"
        ),
        "w"
    ) as f:

        json.dump(
            json_results,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("FINAL LLAMA vs BERT COMPARISON")
    print("=" * 70)

    print(
        table.to_string(
            index=False
        )
    )


def main():

    print("=" * 70)
    print("L3 LIQUIDATION CASCADE")
    print("LLAMA vs BERT EXPERIMENT")
    print("=" * 70)

    print(f"\nDevice: {device}")

    train_loader, test_loader = build_dataset()

    torch.manual_seed(SEED)

    llama = LlamaModel(
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        seq_len
    ).to(device)

    llama_train = train_model(
        llama,
        train_loader,
        "LLAMA"
    )

    llama_results = evaluate_model(
        llama,
        test_loader,
        "LLAMA"
    )

    torch.manual_seed(SEED)

    bert = BERTModel(
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        seq_len
    ).to(device)

    bert_train = train_model(
        bert,
        train_loader,
        "BERT"
    )

    bert_results = evaluate_model(
        bert,
        test_loader,
        "BERT"
    )

    torch.save(
        bert.state_dict(),
        model_save_path_b
    )

    plot_loss(
        llama_train,
        bert_train
    )

    plot_roc(
        llama_results,
        bert_results
    )

    plot_pr(
        llama_results,
        bert_results
    )

    plot_metrics(
        llama_results,
        bert_results
    )

    plot_latency(
        llama_results,
        bert_results
    )

    save_results(
        llama_results,
        bert_results,
        llama_train,
        bert_train
    )

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()