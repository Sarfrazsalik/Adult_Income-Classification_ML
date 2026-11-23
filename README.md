# Adult_Income-Classification_ML


## Problem statement
The objective of this project is to compare five supervised classification
algorithms for predicting whether an individual's annual income is above
$50,000 or not. All models are evaluated on the same stratified 80:20
train-test split using Accuracy, AUC, Precision, Recall, F1-score and MCC.

##  Dataset description
**Dataset:** Adult (Census Income)  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/2/adult

The dataset contains demographic and employment-related attributes such as
age, education, occupation, marital status, working hours and capital gain.
The target column is `income`, with two classes: `<=50K` and `>50K`.
The original data has 14 predictor attributes, a mixture of numerical and
categorical variables, and missing entries represented by `?`.

The script reports the exact record count, duplicate count, missing-value
count and class distribution when it is run. Use those generated values in
your final write-up rather than copying estimates from another source.

## GitHub Repository Link
**   https://github.com/Sarfrazsalik/Adult_Income-Classification_ML.git       **

## Models used and evaluation
The following five models are implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Preprocessing
- Missing numerical values are imputed using the training-set median.
- Missing categorical values are imputed using the training-set most frequent value.
- Categorical features are one-hot encoded.
- Numerical features are standardized.
- Duplicate records are removed before splitting.
- The data is divided into 80% training and 20% testing with stratification.
- `random_state=42` is used for reproducibility.
- Preprocessing is kept inside each model pipeline so it is fitted only on
  the training data and then applied to the test data.

### Model configurations
The exact hyperparameters are documented in the `model/train_models.ipynb` notebook.
The main settings include 1,000 iterations for Logistic Regression, a
maximum depth of 12 for the Decision Tree, 7 neighbours for KNN, Gaussian
Naive Bayes, and 200 trees for Random Forest.

### Results table
The following results are from the current `model/train_models.ipynb` run
(`output/metrics.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8082 | 0.9080 | 0.5660 | 0.8519 | 0.6801 | 0.5744 |
| Decision Tree | 0.8023 | 0.8900 | 0.5565 | 0.8574 | 0.6750 | 0.5682 |
| KNN | 0.8405 | 0.8787 | 0.6862 | 0.6152 | 0.6488 | 0.5474 |
| Naive Bayes | 0.5990 | 0.8173 | 0.3672 | 0.9336 | 0.5271 | 0.3736 |
| Random Forest (Ensemble) | 0.8129 | 0.9159 | 0.5732 | 0.8545 | 0.6861 | 0.5827 |

### Graph representation
The notebook saves `output/model_comparison.png`, which compares all six
metrics for the five models. Include this graph in the final PDF.

![Comparison of Accuracy, AUC, Precision, Recall, F1, and MCC for all five models](outputs/model_comparison.png)

### Observations on model performance
The observations below are based on the current notebook results. Accuracy
alone is not used to select the best model because the target classes are
imbalanced.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good discrimination (AUC 0.9080) and strong Recall (0.8519), but lower Precision (0.5660) means more false positives among predicted `>50K` cases. F1 is 0.6801 and MCC is 0.5744. |
| Decision Tree | Strong Recall (0.8574), but lower Precision (0.5565) and AUC (0.8900). Its F1 (0.6750) and MCC (0.5682) are below Random Forest's. |
| KNN | Highest Accuracy (0.8405) and Precision (0.6862), but lower Recall (0.6152), F1 (0.6488), and MCC (0.5474), indicating it misses more `>50K` cases. |
| Naive Bayes | Highest Recall (0.9336), but low Accuracy (0.5990), Precision (0.3672), F1 (0.5271), and MCC (0.3736); its predictions include many false positives. |
| Random Forest (Ensemble) | Highest AUC (0.9159), F1 (0.6861), and MCC (0.5827), with strong Recall (0.8545), giving it the best overall balance in this run. |
| Overall winner | Random Forest, when selecting by F1. It also has the highest AUC and MCC, while KNN's higher Accuracy comes with substantially lower Recall. |

## How to run
1. Install Python 3.9 or later.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Open `model/train_models.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and select **Run All**. The first run requires an internet connection to download the public UCI data.
5. Review `output/metrics.csv` and `output/model_comparison.png`.

## Repository contents
- `model/train_models.ipynb` - data download, preprocessing, training, evaluation, comparison table, and graph
- `requirements.txt` - Python dependencies
- `test_data.csv` - test data
- `output/` - metrics CSV and comparison graph (created after running)

## Reproducibility
The split uses a fixed random seed (`42`). All models use the same train-test
partition. The metrics and observations in the submitted report must be
based on the output produced by your own execution.
