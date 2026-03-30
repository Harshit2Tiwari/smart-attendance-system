"""
check_accuracy.py
=================
Loads trained model + test data and produces:
  • Per-student accuracy table
  • Confusion matrix heatmap  → models/confusion_matrix.png
  • Model comparison bar chart → models/model_comparison.png
  • Full metrics report       → models/accuracy_report.txt
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(cm, interpolation="nearest",
                   cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right",
                       fontsize=8, color="white")
    ax.set_yticklabels(class_names, fontsize=8, color="white")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black",
                    fontsize=9, fontweight="bold")

    ax.set_title("Confusion Matrix — Smart Attendance System",
                 color="white", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Label", color="#94a3b8", fontsize=10)
    ax.set_ylabel("True Label",      color="#94a3b8", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


def plot_model_comparison(results, save_path):
    names   = [r["name"] for r in results]
    acc     = [r["accuracy"]  * 100 for r in results]
    f1      = [r["f1"]        * 100 for r in results]
    cv_mean = [r["cv_mean"]   * 100 for r in results]

    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    b1 = ax.bar(x - width, acc,     width, label="Val Accuracy",   color="#3b82f6", alpha=0.9)
    b2 = ax.bar(x,          f1,      width, label="F1-Score",       color="#10b981", alpha=0.9)
    b3 = ax.bar(x + width,  cv_mean, width, label="CV Mean",        color="#f59e0b", alpha=0.9)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, color="white", fontsize=9)
    ax.set_ylabel("Score (%)", color="#94a3b8")
    ax.set_ylim(0, 115)
    ax.set_title("Model Comparison — Smart Attendance System",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    ax.yaxis.set_tick_params(labelcolor="white")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.yaxis.grid(True, color="#334155", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


def main():
    print("=" * 55)
    print("  Smart Attendance — Accuracy Check & Report")
    print("=" * 55)

    # Load
    art      = load("dataset/processed/preprocessed.pkl")
    model    = load("models/best_model.pkl")
    pipeline = load("models/pipeline.pkl")
    results  = load("models/all_results.pkl")

    X_te, y_te = art["X_test"], art["y_test"]
    le         = art["label_encoder"]

    y_pred = model.predict(X_te)

    # ── Overall metrics ──────────────────────────────────────────
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)

    print(f"\n[OVERALL TEST METRICS]")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")

    # ── Per-student accuracy ─────────────────────────────────────
    print(f"\n[PER-STUDENT ACCURACY]")
    print(f"  {'Student ID':<10} {'Name':<22} {'Correct':>7} {'Total':>6} {'Acc':>7}")
    print(f"  {'-'*55}")

    for cls_idx, sid in enumerate(le.classes_):
        mask    = (y_te == cls_idx)
        total   = mask.sum()
        correct = (y_pred[mask] == cls_idx).sum()
        name    = pipeline["students"].get(sid, sid)
        pct     = correct / total * 100 if total > 0 else 0
        bar     = "▓" * int(pct // 10) + "░" * (10 - int(pct // 10))
        print(f"  {sid:<10} {name:<22} {correct:>7} {total:>6} {pct:>6.1f}%  {bar}")

    # ── Confusion matrix ─────────────────────────────────────────
    print(f"\n[CONFUSION MATRIX]")
    cm = confusion_matrix(y_te, y_pred)
    print(cm)

    print(f"\n[CLASSIFICATION REPORT]")
    print(classification_report(y_te, y_pred,
                                target_names=le.classes_,
                                zero_division=0))

    # ── Model comparison table ───────────────────────────────────
    print(f"\n[MODEL COMPARISON SUMMARY]")
    print(f"  {'Model':<25} {'Accuracy':>9} {'F1':>9} {'CV Mean':>9}")
    print(f"  {'-'*55}")
    for r in results:
        marker = "  🏆" if r["name"] == pipeline["best_model_name"] else ""
        print(f"  {r['name']:<25} {r['accuracy']*100:>8.2f}% "
              f"{r['f1']*100:>8.2f}% {r['cv_mean']*100:>8.2f}%{marker}")

    # ── Plots ────────────────────────────────────────────────────
    print(f"\n[GENERATING PLOTS]")
    plot_confusion_matrix(cm, list(le.classes_), "models/confusion_matrix.png")
    plot_model_comparison(results, "models/model_comparison.png")

    # ── Text report ──────────────────────────────────────────────
    report_lines = [
        "Smart Attendance System — Accuracy Report",
        "=" * 50,
        f"Best Model    : {pipeline['best_model_name']}",
        f"Test Accuracy : {acc*100:.2f}%",
        f"F1-Score      : {f1*100:.2f}%",
        f"Precision     : {prec*100:.2f}%",
        f"Recall        : {rec*100:.2f}%",
        "",
        "Classification Report:",
        classification_report(y_te, y_pred,
                              target_names=le.classes_,
                              zero_division=0),
    ]
    with open("models/accuracy_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved → models/accuracy_report.txt")

    print(f"\n[SUCCESS] All checks complete!")


if __name__ == "__main__":
    main()
