# Census Income Classifier Explorer

Machine Learning Assignment 2 — M.Tech (AIML/DSE), BITS Pilani WILP

## a. Problem Statement

Predict whether an individual's annual income exceeds **$50,000** based on
demographic and employment attributes collected in a census survey. This is a
**binary classification** problem: `<=50K` vs `>50K`. Six classification
models are trained on the same dataset, evaluated with a common set of
metrics, and compared through an interactive Streamlit application.

## b. Dataset Description

**Dataset:** [UCI Adult / Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/adult)

- **Source:** UCI Machine Learning Repository (extracted from the 1994 US
  Census database by Barry Becker).
- **Instances:** 48,842 total records (32,561 train + 16,281 test, as
  originally split by UCI); after dropping rows with missing values
  (marked `?`), **45,222 instances** remain — well above the assignment's
  500-instance minimum.
- **Features:** 14 raw attributes (6 numeric, 8 categorical) — above the
  assignment's 12-feature minimum:
  - Numeric: `age`, `fnlwgt`, `education_num`, `capital_gain`,
    `capital_loss`, `hours_per_week`
  - Categorical: `workclass`, `education`, `marital_status`, `occupation`,
    `relationship`, `race`, `sex`, `native_country`
- **Target:** `income` — binarized to `1` if `>50K`, else `0`.
- **Preprocessing:** Numeric features are standardized
  (`StandardScaler`); categorical features are one-hot encoded
  (`OneHotEncoder`, unknown categories ignored at inference time). An
  80/20 stratified train/test split is used for model evaluation.
- **Class balance:** ~76% earn `<=50K`, ~24% earn `>50K` (moderately
  imbalanced), which is reflected in the precision/recall trade-offs
  observed below.

`test_data.csv` (in the repo root) contains a 1,000-row held-out sample
drawn from the test split, in raw (untransformed) feature form, for use
with the Streamlit app's upload feature.

## c. GitHub Repository Link

> **TODO:** Replace with your actual repository URL after pushing, e.g.
> `https://github.com/<your-username>/census-income-classifier`

## d. Models Used

Five classifiers (one of them an ensemble) were trained on identical
train/test splits with the same preprocessing pipeline:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier (k=15)
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble, 200 trees)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8450 | 0.9020 | 0.7344 | 0.5870 | 0.6525 | 0.5601 |
| Decision Tree | 0.8473 | 0.8953 | 0.7776 | 0.5379 | 0.6359 | 0.5581 |
| kNN | 0.8401 | 0.8894 | 0.7058 | 0.6088 | 0.6537 | 0.5531 |
| Naive Bayes | 0.6193 | 0.8389 | 0.3877 | 0.9242 | 0.5462 | 0.3891 |
| Random Forest (Ensemble) | 0.8568 | 0.9135 | 0.7907 | 0.5745 | 0.6655 | 0.5892 |

*(Metrics computed on the 20% held-out test split; see `model/train_models.py`
for the exact evaluation code and `model/artifacts/metrics.csv` for the raw
output.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-calibrated baseline (AUC 0.902) given the linear decision boundary. Handles the mostly-linear relationship between income and features like `education_num` and `hours_per_week` well, but slightly under-predicts the minority `>50K` class (Recall 0.59), since a linear boundary can't fully capture interactions such as age × occupation. |
| Decision Tree | Highest Precision (0.7792) among single models — when it predicts `>50K` it is usually right — but Recall drops to 0.5384, the lowest of all non-Naive-Bayes models. A single tree (even depth-limited to 10) tends to fragment the minority class into small, over-specific leaves, trading recall for precision. |
| kNN | Middle-of-the-pack across all metrics; the best Recall (0.6088) of the non-Naive-Bayes models, reflecting that local neighborhoods in scaled feature space capture some of the minority class structure. Performance is sensitive to `k` and to the curse of dimensionality introduced by one-hot encoding 8 categorical columns into a high-dimensional sparse space. |
| Naive Bayes | Clear outlier — Accuracy collapses to 0.6193 and Precision to 0.3877, but Recall is by far the highest (0.9242). Gaussian Naive Bayes assumes numeric features are independent given the class, which is strongly violated here (e.g., `education_num` and `capital_gain` are correlated), causing it to over-predict the `>50K` class. It is the only model where the independence assumption is clearly the bottleneck rather than data volume or class imbalance. |
| Random Forest (Ensemble) | Best model overall on every metric except Recall: highest Accuracy (0.8572), AUC (0.9135), Precision (0.7907), F1 (0.6667), and MCC (0.5903). Averaging 200 decorrelated trees reduces the variance/overfitting a single Decision Tree suffers from, while still capturing non-linear feature interactions that Logistic Regression cannot — at the cost of being the least interpretable model and by far the largest on disk (~33 MB vs. <1 MB for Logistic Regression). |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — it leads on 5 of 6 metrics (Accuracy, AUC, Precision, F1, MCC) and is a close second on Recall, making it the most balanced and reliable choice for this dataset. |

## Project Structure

```
project-folder/
├── app.py                     # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv              # Held-out sample for the app's upload feature
├── data/                      # Raw UCI Adult dataset (adult.data, adult.test)
└── model/
    ├── train_models.py        # Preprocessing, training, evaluation script
    └── artifacts/             # Saved preprocessor + 5 trained models + metrics
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates model/artifacts/ and test_data.csv
streamlit run app.py
```

## Streamlit App Features

- **CSV upload** — upload a test CSV matching the schema of `test_data.csv`
- **Model selection dropdown** — switch between all 5 trained models
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC computed
  live on the uploaded data
- **Confusion matrix, ROC curve, and classification report** — visual and
  tabular breakdown of predictions vs. ground truth
- **Training-time comparison table** — all 6 models side-by-side for reference

## Live Links

- **GitHub Repository:** _TODO — add link_
- **Live Streamlit App:** _TODO — add link after deploying to Streamlit
  Community Cloud_
