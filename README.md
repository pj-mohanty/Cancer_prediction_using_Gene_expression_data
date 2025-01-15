# Cancer Prediction Using Gene Expression Data

## Overview

This project utilizes machine learning techniques to predict cancer types based on gene expression data. The dataset contains gene expression levels for various cancer types, and the objective is to build a predictive model that classifies the cancer type accurately. The analysis includes data exploration, preprocessing, feature selection, model training, and evaluation.

## Table of Contents

1. [Dataset](#dataset)
2. [Dependencies](#dependencies)
3. [Steps in the Analysis](#steps-in-the-analysis)
4. [Feature Selection](#feature-selection)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results](#results)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

---

## Dataset

The dataset used in this project contains:
- **801 samples**.
- **8000 gene expression features**.
- **5 cancer types**:
  - BRCA (300 samples)
  - KIRC (146 samples)
  - LUAD (141 samples)
  - PRAD (136 samples)
  - COAD (78 samples)

---

## Dependencies

The following Python libraries are required:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Steps in the Analysis

### 1. **Data Loading**
The dataset is read directly from Google Drive using `pandas.read_csv`.

### 2. **Data Exploration**
- Shape: `(801, 8001)`
- No missing values detected.
- Distribution of cancer types visualized using bar and pie charts.

### 3. **Data Preprocessing**
- Features (`X`) and target labels (`y`) are separated.
- Labels are encoded using `LabelEncoder`.
- Data is split into training (80%) and testing (20%) subsets.
- Feature values are normalized using `MinMaxScaler` to improve model performance.

---

## Feature Selection

Feature selection is performed using **Mutual Information (MI)** to identify the most relevant features. The top 300 features with the highest MI scores are selected, reducing dimensionality and improving computational efficiency.

---

## Model Training and Evaluation

### Classification Model
- **Random Forest Classifier** with a One-vs-Rest strategy.
- Features selected using MI are used for training and testing.

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

### Results:
- **Accuracy**: 97.17%
- **Precision**: 97.52%
- **Recall**: 97.52%
- **F1 Score**: 97.48%

Confusion matrix and ROC curves provide further insights into model performance for each class.

---

## Usage

1. Clone this repository and download the dataset.
2. Install the dependencies listed in the [Dependencies](#dependencies) section.
3. Update the `file_url` variable with the path to your dataset.
4. Run the analysis notebook or script to reproduce the results.

---

## Conclusion

This analysis demonstrates the potential of machine learning in cancer type prediction using gene expression data. With proper feature selection and preprocessing, the Random Forest Classifier achieved high accuracy, proving its effectiveness in handling high-dimensional and multiclass datasets.

Further improvements may involve experimenting with different feature selection techniques, hyperparameter tuning, or trying other machine learning algorithms.

---

### Author
[Padmaja Mohanty](mailto:padmaja.mohanty@example.com)
