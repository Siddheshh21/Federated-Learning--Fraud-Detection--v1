
import os
import json
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from datetime import datetime as dt
from src.logger import log
from src.config import Cfg

# -----------------------------
# Model performance evaluation
# -----------------------------

def evaluate_model(y_true, y_pred, save=True, tag="global_model", is_attack_scenario=False):
    """
    Evaluate classification performance of the global model.
    y_true: ground-truth labels
    y_pred: predicted probabilities (floats in [0,1])
    is_attack_scenario: boolean indicating if this evaluation is for an attacked model
    """
    y_bin = (y_pred > 0.5).astype(int)
    
    # Calculate base metrics
    base_metrics = {
        "accuracy": float(accuracy_score(y_true, y_bin)),
        "precision": float(precision_score(y_true, y_bin, zero_division=0)),
        "recall": float(recall_score(y_true, y_bin, zero_division=0)),
        "f1": float(f1_score(y_true, y_bin, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_pred))
    }
    
    metrics = base_metrics

    # Confusion matrix
    cm = confusion_matrix(y_true, y_bin)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    if save:
        os.makedirs(f"{Cfg.OUT}/metrics", exist_ok=True)
        path = f"{Cfg.OUT}/metrics/{tag}_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"✓ Saved model metrics to {path}")

    return metrics


# -----------------------------
# Detection evaluation
# -----------------------------

def evaluate_detection_with_bands(y_true, risk_scores, save=True, tag="detection"):
    """
    Evaluate detection performance using review band system.
    y_true: array of 0/1 (0=honest, 1=attacker)
    risk_scores: array of floats in [0,1]
    
    Review band system:
    - score >= 0.80 → block as fraud (high confidence)
    - 0.40 <= score < 0.80 → review/manual check (medium confidence)
    - score < 0.40 → accept (low confidence)
    """
    # Apply review band thresholds
    high_confidence = risk_scores >= 0.80
    review_band = (risk_scores >= 0.40) & (risk_scores < 0.80)
    low_confidence = risk_scores < 0.40
    
    # Binary prediction for metrics (high confidence + review band as positive)
    y_pred = risk_scores >= 0.40
    
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold_low": 0.40,
        "threshold_high": 0.80,
        "high_confidence_predictions": int(sum(high_confidence)),
        "review_band_predictions": int(sum(review_band)),
        "low_confidence_predictions": int(sum(low_confidence))
    }
    
    # Add confusion matrix breakdown by bands
    if y_true is not None:
        high_tp = sum((y_true == 1) & high_confidence)
        high_fp = sum((y_true == 0) & high_confidence)
        review_tp = sum((y_true == 1) & review_band)
        review_fp = sum((y_true == 0) & review_band)
        
        metrics.update({
            "high_confidence_true_positives": int(high_tp),
            "high_confidence_false_positives": int(high_fp),
            "review_band_true_positives": int(review_tp),
            "review_band_false_positives": int(review_fp)
        })

    if save:
        os.makedirs(f"{Cfg.OUT}/metrics", exist_ok=True)
        path = f"{Cfg.OUT}/metrics/{tag}_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"✓ Saved detection metrics with review bands to {path}")

    return metrics


def evaluate_detection(y_true, risk_scores, threshold=None, save=True, tag="detection"):
    """
    Evaluate detection performance given ground-truth attacker labels and risk scores.
    y_true: array of 0/1 (0=honest, 1=attacker)
    risk_scores: array of floats in [0,1]
    threshold: risk threshold for flagging attackers (default = 90th percentile)
    """
    if threshold is None:
        threshold = np.percentile(risk_scores, 95)  # More conservative threshold to reduce false positives

    y_pred = (risk_scores >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold)
    }

    if save:
        os.makedirs(f"{Cfg.OUT}/metrics", exist_ok=True)
        path = f"{Cfg.OUT}/metrics/{tag}_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        log(f"✓ Saved detection metrics to {path}")

    return metrics

def evaluate_attack_impact(training_results: dict[str, Any], detection_results: dict[str, Any] | None) -> dict[str, Any]:
    """
    Evaluate the impact and effectiveness of an attack attempt.
    
    Args:
        training_results: Results from federated training including performance metrics
        detection_results: Results from attack detection including confidence scores (can be None)
    
    Returns:
        Dictionary containing attack impact metrics
    """
    # Extract relevant metrics
    model_metrics = training_results.get("model_metrics", {})
    initial_performance = training_results.get("initial_metrics", {})
    
    # Calculate model performance impact with enhanced attack sensitivity
    initial_acc = initial_performance.get("accuracy", 0)
    initial_f1 = initial_performance.get("f1", 0)
    initial_auc = initial_performance.get("auc", 0)
    
    # Use actual evaluated model metrics (no post-hoc penalties)
    attacked_acc = model_metrics.get("accuracy", 0)
    attacked_f1 = model_metrics.get("f1", 0)
    attacked_auc = model_metrics.get("auc", 0)
    
    performance_impact = {
        "accuracy_change": attacked_acc - initial_acc,
        "f1_change": attacked_f1 - initial_f1,
        "auc_change": attacked_auc - initial_auc
    }
    
    # Calculate attack success metrics
    attack_metrics = {
        "attack_success_rate": 0.0,  # Default to 0
        "detection_accuracy": 0.0,
        "false_positive_rate": 0.0
    }
    
    if detection_results and "detection_metrics" in detection_results:
        det_metrics = detection_results["detection_metrics"]
        attack_metrics.update({
            "detection_accuracy": det_metrics.get("accuracy", 0.0),
            "false_positive_rate": det_metrics.get("fp_rate", 0.0)
        })
    
    # Calculate attack success rate based on attack type with enhanced impact
    attack_type = training_results.get("attack_type", "unknown")
    if attack_type == "label_flip":
        # Success measured by combined impact on accuracy and F1
        acc_impact = abs(performance_impact["accuracy_change"])
        f1_impact = abs(performance_impact["f1_change"])
        attack_metrics["attack_success_rate"] = max(acc_impact * 1.5, f1_impact * 1.3)  # Amplify impact
    
    elif attack_type == "backdoor":
        # Success measured by backdoor success rate and model degradation
        backdoor_rate = training_results.get("backdoor_success_rate", 0.0)
        model_degradation = abs(performance_impact["accuracy_change"]) + abs(performance_impact["f1_change"])
        attack_metrics["attack_success_rate"] = max(backdoor_rate * 1.2, model_degradation)
    
    elif attack_type == "byzantine":
        # Enhanced Byzantine impact calculation
        model_divergence = training_results.get("model_divergence", 0.0)
        byzantine_impact = training_results.get("byzantine_impact", 0.0)
        update_magnitude = training_results.get("update_magnitude", 0.0)
        
        # Stronger impact factors
        divergence_factor = min(1.0, model_divergence * 3.0)  # Increased divergence impact
        magnitude_factor = min(1.0, update_magnitude / 500.0)  # More sensitive to updates
        performance_factor = max(
            abs(performance_impact["accuracy_change"]) * 2.0,
            abs(performance_impact["f1_change"]) * 1.5,
            abs(performance_impact["auc_change"]) * 3.0
        )
        
        attack_metrics["attack_success_rate"] = max(
            divergence_factor,
            byzantine_impact * 1.5,
            magnitude_factor * 0.8,
            performance_factor
        )
    
    elif attack_type == "scaling":
        # Enhanced scaling attack impact
        scaling_dominance = training_results.get("scaling_dominance", 0.0)
        performance_degradation = max(
            abs(performance_impact["accuracy_change"]) * 1.5,
            abs(performance_impact["f1_change"]) * 1.3
        )
        attack_metrics["attack_success_rate"] = max(scaling_dominance * 1.4, performance_degradation)
    
    elif attack_type == "free_ride":
        # Enhanced free-ride impact with model degradation
        contribution_avoidance = training_results.get("contribution_avoidance", 0.0)
        auc_impact = abs(performance_impact["auc_change"]) * 2.0
        attack_metrics["attack_success_rate"] = max(contribution_avoidance * 1.3, auc_impact)
    
    elif attack_type == "sybil":
        # Enhanced Sybil attack impact
        sybil_influence = training_results.get("sybil_influence", 0.0)
        model_impact = max(
            abs(performance_impact["accuracy_change"]) * 1.8,
            abs(performance_impact["f1_change"]) * 1.5,
            abs(performance_impact["auc_change"]) * 2.0
        )
        attack_metrics["attack_success_rate"] = max(sybil_influence * 1.5, model_impact)
    
    # Combine all metrics
    return {
        "model_performance_impact": performance_impact,
        "attack_success_rate": attack_metrics["attack_success_rate"],
        "detection_accuracy": attack_metrics["detection_accuracy"],
        "false_positive_rate": attack_metrics["false_positive_rate"],
        "attack_confidence": detection_results.get("confidence", 0.0) if detection_results else 0.0,
        "attack_type_detected": detection_results.get("attack_types", []) if detection_results else [],
        "triggered_rules": detection_results.get("triggered_rules", []) if detection_results else []
    }


# -----------------------------
# Visualization helpers
# -----------------------------

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    fig.colorbar(im)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_roc_curve(y_true, y_pred, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_risk_distribution(risk_scores, y_true=None, save_path=None):
    fig, ax = plt.subplots()
    if y_true is not None:
        ax.hist([risk_scores[y_true==0], risk_scores[y_true==1]],
                bins=20, stacked=True, label=["Honest","Attacker"], alpha=0.7)
        ax.legend()
    else:
        ax.hist(risk_scores, bins=20, alpha=0.7)
    ax.set_title("Risk Score Distribution")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

