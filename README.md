# 🌸 Flower Classification — Iris Dataset
 
A complete end-to-end machine learning pipeline on the classic **Iris dataset** — covering exploratory data analysis, visualization, model training, evaluation, and a command-line predictor for new flower measurements.
 
---
 
## 📋 Overview
 
This project demonstrates a full ML workflow:
 
1. **Load** the Iris dataset into a clean, human-readable DataFrame
2. **Explore** the data with summary statistics and correlation analysis
3. **Visualize** feature relationships (pair plots, box plots, heatmaps)
4. **Train** three classifiers — Logistic Regression, Decision Tree, and KNN
5. **Tune** the KNN model with cross-validation and `GridSearchCV`
6. **Compare** model accuracy and inspect confusion matrices
7. **Predict** species for new flower measurements via an interactive CLI
---
 
## 🌼 Dataset
 
The [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) contains 150 samples of iris flowers across 3 species, with 4 measured features each:
 
| Feature | Description |
|---|---|
| `sepal length (cm)` | Length of the sepal |
| `sepal width (cm)`  | Width of the sepal |
| `petal length (cm)` | Length of the petal |
| `petal width (cm)`  | Width of the petal |
 
**Target classes:** `setosa`, `versicolor`, `virginica` (50 samples each)
 
---

## 🗂️ Project Structure
 
```
flower-classification/
│
├── flower_class.py          # Main script — EDA, training, evaluation, CLI
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
└── outputs/                  # Auto-generated plots
    ├── 01_pair_plot.png
    ├── 02_box_plots.png
    ├── 03_correlation_heatmap.png
    ├── 04_knn_gridsearch.png
    ├── 05_accuracy_comparison.png
    ├── 06_decision_tree.png
    └── 07_confusion_matrices.png
```
 

 
## ⚙️ Installation
 
```bash
# Clone the repository
git clone https://github.com/<your-username>/flower-classification.git
cd flower-classification
 
# (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
 
# Install dependencies
pip install -r requirements.txt
```
 
---
 
## ▶️ Usage
 
**Run the full pipeline** (EDA → visualizations → training → evaluation):
 
```bash
python flower_class.py
```
 
**Run with the interactive species predictor:**
 
```bash
python flower_class.py --cli
```
 
You'll be prompted to enter four measurements:
 
```
Enter 4 space-separated values in cm:
Order: sepal length (cm)  |  sepal width (cm)  |  petal length (cm)  |  petal width (cm)
Example: 5.1 3.5 1.4 0.2
 
>> 6.5 2.8 7.1 1.5
```
 
The script outputs predictions and class probabilities from **Logistic Regression**, **Decision Tree**, and **KNN**.
 
---


 
## 🧠 Models Used
 
| Model | Notes |
|---|---|
| **Logistic Regression** | Wrapped in a `Pipeline` with `StandardScaler` |
| **Decision Tree** | `max_depth=4`, visualized as a tree diagram |
| **K-Nearest Neighbors** | Tuned via `GridSearchCV` (k = 1–15) with 5-fold stratified cross-validation |
 
---
 
## 🔍 Key Findings
 
- **Setosa** is always perfectly separable from the other two species due to distinct petal dimensions.
- **Versicolor** and **Virginica** show some feature overlap, which is the primary source of misclassification across all models.
- All three models typically achieve **90%+ accuracy** on this dataset, given its small size and well-separated classes.
---
 
## 🛠️ Tech Stack
 
- Python 3.x
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling, evaluation, and pipelines
---
 

 
## 🙌 Acknowledgements
 
- Dataset originally introduced by **Ronald Fisher** (1936)
