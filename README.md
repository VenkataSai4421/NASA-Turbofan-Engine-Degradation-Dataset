# NASA-Turbofan-Engine-Degradation-Dataset
Aerospace Engine Failure Prediction with Random Forest
This project uses a Random Forest Classifier to predict engine failure in the NASA CMAPSS Turbofan Engine Degradation Dataset (FD001). The goal is to classify whether an engine’s Remaining Useful Life (RUL) is ≤ 30 cycles, enabling predictive maintenance in aerospace applications.
Dataset
The dataset is sourced from NASA’s Prognostics Data Repository (CMAPSS Dataset) and is also available on Kaggle (NASA CMAPSS). We use the train_FD001.txt subset, which includes:

100 engines run to failure.
Features: 3 operational settings and 21 sensor measurements.
Target: Binary classification (1 if RUL ≤ 30 cycles, 0 otherwise).

Requirements
To run the code, install the following Python libraries (pre-installed in Kaggle):

pandas
numpy
scikit-learn
matplotlib
seaborn

Install locally using:
pip install pandas numpy scikit-learn matplotlib seaborn

Project Structure

eda-random-forest.ipynb: Jupyter notebook with data loading, EDA, preprocessing, Random Forest training, and evaluation.
README.md: This file.

How to Run
In Kaggle

Open Kaggle Notebook:
Create a new notebook or use an existing one.
Ensure the CMAPSS dataset is attached: /kaggle/input/nasa-turbofan-engine-degradation-dataset/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.


Copy Code:
Copy the code from eda-random-forest.ipynb into a notebook cell.


Verify Dataset Path:
Check that train_FD001.txt is at /kaggle/input/nasa-turbofan-engine-degradation-dataset/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData/train_FD001.txt.
If needed, list files:import os
print(os.listdir("/kaggle/input/nasa-turbofan-engine-degradation-dataset/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData"))




Run:
Execute the notebook to perform EDA, train the Random Forest Classifier, and view results.



Locally

Download Dataset:
Get the CMAPSS dataset from NASA or Kaggle.
Extract to a folder (e.g., CMAPSSData/).


Update Path:
In the notebook, change the data_path to your local train_FD001.txt (e.g., CMAPSSData/train_FD001.txt).


Run:
Open eda-random-forest.ipynb in Jupyter and execute.



Results
The Random Forest Classifier was trained on the FD001 dataset with a 90/10 train-test split. Performance metrics:

Accuracy: 95.4%
Classification Report:

              precision    recall  f1-score   support
           0       0.97      0.98      0.97      1747
           1       0.87      0.82      0.85       317
    accuracy                           0.95      2064
   macro avg       0.92      0.90      0.91      2064
weighted avg       0.95      0.95      0.95      2064


Confusion Matrix:

[[1709   38]
 [  57  260]]

Key insights:

High accuracy (95.4%) indicates strong predictive performance.
Good precision/recall for class 0 (RUL > 30), with slightly lower recall for class 1 (RUL ≤ 30) due to class imbalance.
Feature importance plots (in notebook) highlight key sensors driving predictions.

Future Work

Evaluate on the test set (test_FD001.txt) using RUL_FD001.txt.
Try regression to predict exact RUL values.
Perform additional EDA (e.g., sensor correlations, time-series trends).
Optimize the model with hyperparameter tuning (e.g., n_estimators, max_depth).
Explore other models (e.g., XGBoost, LSTM for time-series).

License
This project is licensed under the MIT License.
