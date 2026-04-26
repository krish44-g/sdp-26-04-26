"""
Training script for SpineAI.
Run: python -m training.train --annotation_file data/annotations.json
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import json
from pathlib import Path
from model.pipeline import SpineAIModel
from training.dataset import SpineDataset
from training.metrics import compute_metrics


def combined_loss(outputs, batch, heatmap_weight=1.0, cls_weight=2.0, severity_weight=0.5):
    """Combined loss: heatmap MSE + BCE classification + severity MAE."""
    # Heatmap loss
    hm_loss = nn.functional.mse_loss(outputs["heatmaps"], batch["heatmaps"])
    # Classification loss
    cls_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["logits"], batch["labels"]
    )
    # Severity loss (only for positive classes)
    mask = batch["labels"] > 0
    sev_loss = torch.tensor(0.0)
    if mask.any():
        sev_pred = torch.sigmoid(outputs["logits"])
        sev_loss = nn.functional.l1_loss(sev_pred[mask], batch["severity"][mask])

    total = heatmap_weight * hm_loss + cls_weight * cls_loss + severity_weight * sev_loss
    return total, {"hm": hm_loss.item(), "cls": cls_loss.item(), "sev": sev_loss.item()}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = SpineDataset(args.annotation_file, split="train")
    val_size = int(0.15 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SpineAIModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    Path("model/weights").mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch["image"], batch["ethnicity_idx"], batch["raw_ratios"])
            loss, loss_dict = combined_loss(outputs, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch["image"], batch["ethnicity_idx"], batch["raw_ratios"])
                all_preds.append(outputs["probabilities"].cpu())
                all_labels.append(batch["labels"].cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        metrics = compute_metrics(preds, labels)
        val_f1 = metrics["macro_f1"]

        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val F1: {val_f1:.4f} | Val AUC: {metrics['macro_auc']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "model/weights/posturenet.pth")
            print(f"  ✓ Saved best model (F1={val_f1:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", default="data/annotations.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
