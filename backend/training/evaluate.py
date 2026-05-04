"""
Full evaluation script for SpineAI.
Prints per-class and macro metrics, confusion matrix, PCKh@0.5, and exports CSV.

Automatically detects whether the weights file is from TransferSpineModel
(transfer_model.pth) or SpineAIModel (posturenet.pth) and loads accordingly.

Usage:
  # Full PostureNet model:
  python -m training.evaluate \
    --annotation_file data/annotations.json \
    --weights model/weights/posturenet.pth \
    --export_csv results/metrics.csv

  # Transfer learning model:
  python -m training.evaluate \
    --annotation_file data/annotations.json \
    --weights model/weights/transfer_model.pth \
    --export_csv results/metrics.csv
"""
import argparse
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix,
)
from training.metrics import pckh_at_05, severity_mae
from config import settings

CLASSES = settings.DEFORMITY_CLASSES


def load_model(weights_path: str, device: torch.device):
    """
    Detect which model architecture the weights belong to and load it correctly.
    - transfer_model.pth → TransferSpineModel (ResNet-50 backbone)
    - posturenet.pth     → SpineAIModel (custom PostureNet backbone)
    """
    is_transfer = os.path.basename(weights_path) == "transfer_model.pth"

    if is_transfer:
        from training.transfer_learn import TransferSpineModel
        model = TransferSpineModel(num_classes=len(CLASSES), freeze_backbone=False)
        print(f"  Architecture: TransferSpineModel (ResNet-50)")
    else:
        from model.pipeline import SpineAIModel
        model = SpineAIModel()
        print(f"  Architecture: SpineAIModel (PostureNet)")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  ✓ Weights loaded from {weights_path}")
    else:
        print(f"  ⚠ No weights found at {weights_path} — using random init (for testing only)")

    return model.to(device)


def run_inference(model, loader, device, is_transfer: bool):
    """
    Run inference over the dataloader.
    TransferSpineModel only returns logits (no keypoints/heatmaps),
    so keypoint metrics are skipped for transfer model evaluation.
    """
    all_probs, all_labels, all_severity = [], [], []
    all_pred_kps, all_gt_kps = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if is_transfer:
                # TransferSpineModel: forward takes only the image tensor
                logits = model(batch["image"])
                probs = torch.sigmoid(logits)
                # No keypoint output — use zeros so pckh_at_05 returns 0.0
                fake_kps = torch.zeros_like(batch["keypoints"])
                all_pred_kps.append(fake_kps.cpu().numpy())
            else:
                # SpineAIModel: full pipeline
                outputs = model(batch["image"], batch["ethnicity_idx"], batch["raw_ratios"])
                probs = outputs["probabilities"]
                all_pred_kps.append(outputs["keypoints"].cpu().numpy())

            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            all_severity.append(batch["severity"].cpu().numpy())
            all_gt_kps.append(batch["keypoints"].cpu().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_severity),
        np.concatenate(all_pred_kps),
        np.concatenate(all_gt_kps),
    )


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SpineAI Evaluate] Device: {device}")
    print(f"[SpineAI Evaluate] Weights: {args.weights}")
    print(f"[SpineAI Evaluate] Annotation: {args.annotation_file}\n")

    is_transfer = os.path.basename(args.weights) == "transfer_model.pth"

    model = load_model(args.weights, device)
    model.eval()

    from training.dataset import SpineDataset
    dataset = SpineDataset(args.annotation_file, split="val")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    probs, labels, sevs, pred_kp, gt_kp = run_inference(model, loader, device, is_transfer)
    preds = (probs >= 0.5).astype(int)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    per_f1  = f1_score(labels, preds, average=None, zero_division=0)
    per_pre = precision_score(labels, preds, average=None, zero_division=0)
    per_rec = recall_score(labels, preds, average=None, zero_division=0)
    try:
        per_auc = roc_auc_score(labels, probs, average=None).tolist()
    except ValueError:
        per_auc = [0.0] * len(CLASSES)

    # ── Macro/weighted averages ───────────────────────────────────────────────
    macro_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    w_f1      = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_pre = precision_score(labels, preds, average="macro", zero_division=0)
    macro_rec = recall_score(labels, preds, average="macro", zero_division=0)
    try:
        macro_auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        macro_auc = 0.0

    # ── Keypoint + severity metrics ───────────────────────────────────────────
    # PCKh is only meaningful for the full SpineAIModel which has a keypoint head.
    pckh        = pckh_at_05(pred_kp, gt_kp) if not is_transfer else float("nan")
    sev_mae_val = severity_mae(probs, sevs, labels)

    # ── Print results ─────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print("  SpineAI Evaluation Results")
    print("=" * W)
    print(f"  {'Class':<20} {'F1':>6}  {'Precision':>10}  {'Recall':>8}  {'AUC-ROC':>8}")
    print("-" * W)
    rows = []
    for i, cls in enumerate(CLASSES):
        f1_val  = per_f1[i]
        pre_val = per_pre[i]
        rec_val = per_rec[i]
        auc_val = per_auc[i]
        bar = "█" * int(f1_val * 20) + "░" * (20 - int(f1_val * 20))
        print(f"  {cls:<20} {f1_val:>6.4f}  {pre_val:>10.4f}  {rec_val:>8.4f}  {auc_val:>8.4f}  {bar}")
        rows.append({
            "class": cls,
            "f1": round(f1_val, 4),
            "precision": round(pre_val, 4),
            "recall": round(rec_val, 4),
            "auc": round(auc_val, 4),
        })
    print("-" * W)
    print(f"  {'Macro Average':<20} {macro_f1:>6.4f}  {macro_pre:>10.4f}  {macro_rec:>8.4f}  {macro_auc:>8.4f}")
    print(f"  {'Weighted Average':<20} {w_f1:>6.4f}")
    print("=" * W)

    if is_transfer:
        print(f"\n  PCKh@0.5: N/A (TransferSpineModel has no keypoint head)")
    else:
        print(f"\n  PCKh@0.5 (keypoint accuracy): {pckh:.4f}")
    print(f"  Severity MAE:                 {sev_mae_val:.4f}")
    print(f"\n  Samples evaluated: {len(labels)}")
    print("=" * W)

    # ── Confusion matrix (per class, binary) ──────────────────────────────────
    print("\n  Per-class Confusion Matrix (TP/FP/TN/FN):")
    print(f"  {'Class':<20} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("  " + "-" * 40)
    for i, cls in enumerate(CLASSES):
        unique_vals = np.unique(labels[:, i])
        if len(unique_vals) > 1:
            cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        print(f"  {cls:<20} {tp:>5} {fp:>5} {tn:>5} {fn:>5}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    if args.export_csv:
        os.makedirs(os.path.dirname(args.export_csv) or ".", exist_ok=True)
        with open(args.export_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "f1", "precision", "recall", "auc"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({
                "class": "MACRO",
                "f1": round(macro_f1, 4),
                "precision": round(macro_pre, 4),
                "recall": round(macro_rec, 4),
                "auc": round(macro_auc, 4),
            })
        print(f"\n  ✓ Metrics exported to: {args.export_csv}")

    print("\n  Done.\n")
    return {
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "pckh": pckh,
        "sev_mae": sev_mae_val,
        "per_class": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", default="data/annotations.json")
    parser.add_argument("--weights", default="model/weights/posturenet.pth")
    parser.add_argument("--export_csv", default="results/metrics.csv")
    evaluate(parser.parse_args())
