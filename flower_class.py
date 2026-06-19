"""
============================================================
  Iris Flower Classification
============================================================
* Requirements:
      - EDA + feature-pair visualisations
      - Logistic Regression vs Decision Tree accuracy comparison
      - Confusion matrix + misclassification interpretation
      - CLI to predict species for new inputs

Runs end-to-end:  python flower_class.py
Interactive CLI:  python flower_class.py --cli

Output PNGs are saved in the SAME folder as this script.
============================================================
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Output directory = same folder as this script (works on any OS)
# ------------------------------------------------------------------
OUT_DIR = Path(__file__).parent         
OUT_DIR.mkdir(parents=True, exist_ok=True)

def out(filename: str) -> str:
    """Return full cross-platform path for an output file."""
    return str(OUT_DIR / filename)


# ==================================================================
# 1.  LOAD DATASET
# ==================================================================

def load_data():
    
    iris = datasets.load_iris()

    # Build the human-readable DataFrame (Video 1)
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names,      # 'sepal length (cm)', etc.
    )
    df["target"]  = iris.target          # numeric  0 / 1 / 2
    df["species"] = df["target"].map(    # human-readable label
        {i: name for i, name in enumerate(iris.target_names)}
    )

    X = df[iris.feature_names]
    y = df["target"]

    return df, X, y, iris.feature_names, iris.target_names


# ==================================================================
# 2.  EDA
# ==================================================================

def eda(df, target_names):
    """Print exploratory data-analysis summary."""
    print("\n" + "=" * 58)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 58)

    print("\n-- Dataset shape --")
    print(f"  Rows x Columns : {df.shape}")

    print("\n-- First 5 rows (human-readable DataFrame) --")
    print(df.head().to_string())

    print("\n-- Statistical summary --")
    features = [c for c in df.columns if c not in ("target", "species")]
    print(df[features].describe().round(2).to_string())

    print("\n-- Class distribution --")
    counts = df["species"].value_counts().sort_index()
    for name, cnt in counts.items():
        print(f"  {name:12s}: {cnt} samples")

    print("\n-- Missing values --")
    print(f"  {df.isnull().sum().sum()} missing values total")

    print("\n-- Correlation matrix (features only) --")
    print(df[features].corr().round(3).to_string())


# ==================================================================
# 3.  VISUALISATIONS
#     (a) Pair plot  - feature relationships by species
#     (b) Box plots  - per-class distributions
#     (c) Heatmap    - correlation matrix
# ==================================================================

def visualise(df, target_names):
    """Generate and save three visualisation figures."""
    features = [c for c in df.columns if c not in ("target", "species")]

    # (a) Pair plot
    print("\n[VIZ] Generating pair plot ...")
    sns.set_theme(style="ticks", font_scale=0.9)
    pg = sns.pairplot(
        df.drop(columns=["target"]),
        hue="species",
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 40},
        palette={
            "setosa":     "#4C72B0",
            "versicolor": "#DD8452",
            "virginica":  "#55A868",
        },
    )
    pg.figure.suptitle("Iris - Feature Pair Plot (all 3 classes)", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out("01_pair_plot.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {out('01_pair_plot.png')}")

    # (b) Box plots
    print("[VIZ] Generating box plots ...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Feature Distribution by Species", fontsize=14, fontweight="bold")
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, feat in zip(axes.ravel(), features):
        data_by_class = [
            df[df["species"] == name][feat].values for name in target_names
        ]
        bp = ax.boxplot(
            data_by_class, patch_artist=True, notch=False,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(target_names, fontsize=9)
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel("cm")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out("02_box_plots.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {out('02_box_plots.png')}")

    # (c) Correlation heatmap
    print("[VIZ] Generating correlation heatmap ...")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out("03_correlation_heatmap.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"       Saved -> {out('03_correlation_heatmap.png')}")


# ==================================================================
# 4.  KNN - Pipeline + cross-validation + GridSearchCV
#     
# ==================================================================

def knn_pipeline(X, y, target_names):
    
    print("\n" + "=" * 58)
    print("  KNN  -  PIPELINE + CROSS-VALIDATION + GridSearchCV")
    print("=" * 58)

    # Build pipeline: scaler -> KNN
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier()),
    ])

    # 5-fold stratified cross-validation (default k=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(knn_pipe, X, y, cv=cv, scoring="accuracy")
    print(f"\n  KNN (k=5) - 5-fold CV scores : {np.round(cv_scores, 3)}")
    print(f"  Mean +/- Std                  : "
          f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # GridSearchCV: find the best k (1 to 15)
    param_grid = {"knn__n_neighbors": list(range(1, 16))}
    grid = GridSearchCV(
        knn_pipe, param_grid, cv=cv,
        scoring="accuracy", return_train_score=True,
    )
    grid.fit(X, y)

    best_k   = grid.best_params_["knn__n_neighbors"]
    best_acc = grid.best_score_
    print(f"\n  GridSearchCV best k           : {best_k}")
    print(f"  Best CV accuracy              : {best_acc:.4f}")

    # Plot CV accuracy vs k
    results  = grid.cv_results_
    k_values = [p["knn__n_neighbors"] for p in results["params"]]
    mean_acc = results["mean_test_score"]
    std_acc  = results["std_test_score"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, mean_acc, "o-", color="#4C72B0",
            linewidth=2, label="CV accuracy")
    ax.fill_between(
        k_values, mean_acc - std_acc, mean_acc + std_acc,
        alpha=0.2, color="#4C72B0",
    )
    ax.axvline(best_k, color="#DD8452", linestyle="--",
               linewidth=1.5, label=f"Best k={best_k}")
    ax.set_xlabel("Number of Neighbours (k)", fontsize=11)
    ax.set_ylabel("CV Accuracy", fontsize=11)
    ax.set_title("KNN - GridSearchCV: Accuracy vs k",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out("04_knn_gridsearch.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved -> {out('04_knn_gridsearch.png')}")

    # Return the best fitted pipeline
    return grid.best_estimator_, best_k


# ==================================================================
# 5.  TRAIN LOGISTIC REGRESSION + DECISION TREE + KNN
#     Compare all three on the same hold-out test set
# ==================================================================

def train_and_evaluate(X, y, target_names, best_knn_pipe):
    """
    Train all three models and compare accuracy.
    LR and KNN use Pipelines with StandardScaler.
    DT does not need scaling.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y,
    )

    # Logistic Regression (Pipeline)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=500, random_state=42)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_acc  = accuracy_score(y_test, lr_pred)

    # Decision Tree
    dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
    dtree.fit(X_train, y_train)
    dt_pred = dtree.predict(X_test)
    dt_acc  = accuracy_score(y_test, dt_pred)

    # KNN (best pipeline from GridSearchCV)
    best_knn_pipe.fit(X_train, y_train)
    knn_pred = best_knn_pipe.predict(X_test)
    knn_acc  = accuracy_score(y_test, knn_pred)

    # Print comparison
    print("\n" + "=" * 58)
    print("  MODEL ACCURACY COMPARISON  (hold-out test set)")
    print("=" * 58)
    bar = lambda acc: "#" * int(acc * 32)
    print(f"\n  Logistic Regression : {lr_acc*100:6.2f}%  {bar(lr_acc)}")
    print(f"  Decision Tree       : {dt_acc*100:6.2f}%  {bar(dt_acc)}")
    print(f"  KNN (best k)        : {knn_acc*100:6.2f}%  {bar(knn_acc)}")

    print("\n-- Logistic Regression - Classification Report --")
    print(classification_report(y_test, lr_pred, target_names=target_names))

    print("-- Decision Tree - Classification Report --")
    print(classification_report(y_test, dt_pred, target_names=target_names))

    print("-- KNN - Classification Report --")
    print(classification_report(y_test, knn_pred, target_names=target_names))

    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    model_names = ["Logistic\nRegression", "Decision\nTree", "KNN\n(best k)"]
    accs       = [lr_acc, dt_acc, knn_acc]
    bar_colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(
        model_names, accs, color=bar_colors,
        width=0.45, edgecolor="white", linewidth=1.2,
    )
    for bar_, acc in zip(bars, accs):
        ax.text(
            bar_.get_x() + bar_.get_width() / 2, acc + 0.005,
            f"{acc*100:.1f}%", ha="center", va="bottom",
            fontweight="bold", fontsize=11,
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Model Accuracy Comparison (test set)",
                 fontsize=13, fontweight="bold")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out("05_accuracy_comparison.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved -> {out('05_accuracy_comparison.png')}")

    # Decision Tree structure plot
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(
        dtree,
        feature_names=list(X.columns),
        class_names=target_names,
        filled=True, rounded=True, fontsize=9, ax=ax,
    )
    ax.set_title("Decision Tree Structure", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out("06_decision_tree.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved -> {out('06_decision_tree.png')}")

    return {
        "lr_pipe"  : lr_pipe,
        "dtree"    : dtree,
        "knn_pipe" : best_knn_pipe,
        "X_test"   : X_test,
        "y_test"   : y_test,
        "lr_pred"  : lr_pred,
        "dt_pred"  : dt_pred,
        "knn_pred" : knn_pred,
    }


# ==================================================================
# 6.  CONFUSION MATRICES  (all three models)
# ==================================================================

def plot_confusion_matrices(results, target_names):
    """Plot confusion matrices for all three models and interpret them."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    model_preds = [
        ("Logistic Regression", results["lr_pred"]),
        ("Decision Tree",       results["dt_pred"]),
        ("KNN (best k)",        results["knn_pred"]),
    ]

    for ax, (name, y_pred) in zip(axes, model_preds):
        cm   = confusion_matrix(results["y_test"], y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=target_names,
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted label", fontsize=9)
        ax.set_ylabel("True label", fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out("07_confusion_matrices.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved -> {out('07_confusion_matrices.png')}")

    # Narrative interpretation
    print("\n" + "=" * 58)
    print("  CONFUSION MATRIX INTERPRETATION")
    print("=" * 58)
    for name, y_pred in model_preds:
        cm  = confusion_matrix(results["y_test"], y_pred)
        err = int(np.sum(cm) - np.trace(cm))
        print(f"\n  {name}:")
        print(f"    Correct / Total : {int(np.trace(cm))} / {int(np.sum(cm))}")
        print(f"    Misclassified   : {err}")
        if err > 0:
            for i in range(len(target_names)):
                for j in range(len(target_names)):
                    if i != j and cm[i, j] > 0:
                        print(f"      * {cm[i,j]:2d} x '{target_names[i]}'"
                              f" predicted as '{target_names[j]}'")
            print("    -> Setosa is always perfectly separable.")
            print("       Versicolor & Virginica share overlapping petal")
            print("       dimensions, making them the main source of confusion.")
        else:
            print("    -> Perfect classification on this test split")


# ==================================================================
# 7.  CLI  -  PREDICT SPECIES FOR NEW INPUT
# ==================================================================

def predict_new(lr_pipe, dtree, knn_pipe, target_names, feature_names):
    """
    Interactive CLI: enter 4 measurements and get predictions
    from all three models with a probability breakdown.
    """
    print("\n" + "=" * 58)
    print("  IRIS SPECIES PREDICTOR  (type 'quit' to exit)")
    print("=" * 58)
    print("  Enter 4 space-separated values in cm:")
    print("  Order:", "  |  ".join(feature_names))
    print("  Example: 5.1 3.5 1.4 0.2\n")

    while True:
        try:
            raw = input("  >> ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye!\n")
                break

            vals = [float(v) for v in raw.split()]
            if len(vals) != 4:
                print(f"  Expected 4 values, got {len(vals)}. Try again.\n")
                continue

            arr = np.array([vals])

            preds = {
                "Logistic Regression": (
                    lr_pipe.predict(arr)[0],
                    lr_pipe.predict_proba(arr)[0],
                ),
                "Decision Tree": (
                    dtree.predict(arr)[0],
                    dtree.predict_proba(arr)[0],
                ),
                "KNN (best k)": (
                    knn_pipe.predict(arr)[0],
                    knn_pipe.predict_proba(arr)[0],
                ),
            }

            print()
            for model_name, (cls, proba) in preds.items():
                dashes = "-" * (36 - len(model_name))
                print(f"  +-- {model_name} {dashes}")
                print(f"  |   Predicted species : {target_names[cls].upper()}")
                for i, p in enumerate(proba):
                    bar = "|" * int(p * 20)
                    print(f"  |     {target_names[i]:12s}: {p*100:5.1f}%  {bar}")
                print(f"  +{'-'*42}")
            print()

        except ValueError:
            print("  Please enter numeric values only (e.g. 5.1 3.5 1.4 0.2)\n")
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!\n")
            break


# ==================================================================
# MAIN
# ==================================================================

def main():

    # 1. Load - human-readable DataFrame 
    df, X, y, feature_names, target_names = load_data()

    # 2. EDA
    eda(df, target_names)

    # 3. Visualise: pair plot, box plots, heatmap
    visualise(df, target_names)

    # 4. KNN with Pipeline + CV + GridSearchCV 
    best_knn_pipe, best_k = knn_pipeline(X, y, target_names)

    # 5. Train LR, DT, KNN and compare accuracy
    results = train_and_evaluate(X, y, target_names, best_knn_pipe)

    # 6. Confusion matrices + interpretation
    plot_confusion_matrices(results, target_names)

    # 7. Demo prediction 
    print("\n" + "=" * 58)
    print("  DEMO PREDICTION  [ 6.5, 2.8, 7.1, 1.5 ]")
    print("=" * 58)
    demo = np.array([[6.5, 2.8, 7.1, 1.5]])
    for name, model in [
        ("Logistic Regression", results["lr_pipe"]),
        ("Decision Tree",       results["dtree"]),
        (f"KNN (k={best_k})",   results["knn_pipe"]),
    ]:
        pred = model.predict(demo)[0]
        print(f"  {name:22s} -> {target_names[pred]}")

    # 8. Interactive CLI (--cli flag or interactive terminal)
    if "--cli" in sys.argv or (len(sys.argv) == 1 and sys.stdin.isatty()):
        predict_new(
            results["lr_pipe"],
            results["dtree"],
            results["knn_pipe"],
            list(target_names),
            list(feature_names),
        )


if __name__ == "__main__":
    main()
