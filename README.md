# customer-churn-model-using-machine-learning
This is a customer churn prediction model built using machine learning (decision tree, random forest and XGBoost models)

Customer Churn Prediction using Machine Learning

This project builds a machine learning model to predict customer churn using the Telco Customer Churn dataset
.
It applies preprocessing, class balancing, and multiple ML algorithms to identify customers likely to leave a telecom service provider.

📦 Features

Automatic dataset download via Kaggle API.

Data preprocessing: handling missing values, encoding categorical features, scaling.

SMOTE oversampling to address class imbalance.

Multiple models tested:

Decision Tree

Random Forest

XGBoost

Model evaluation with accuracy, confusion matrix, classification report.

Best model saved as a pickle file for reuse.

⚙️ Setup

Clone the repository

git clone https://github.com/yourusername/customer-churn-ml.git
cd customer-churn-ml


Install dependencies

pip install -r requirements.txt


Kaggle API setup

Download your Kaggle API key from Kaggle → Account → Create API Token
.

Place the downloaded kaggle.json in:

Linux/Mac → ~/.kaggle/kaggle.json

Windows → C:\Users\<username>\.kaggle\kaggle.json

📂 Dataset

The dataset is downloaded automatically with:

import opendatasets as od
od.download("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")


File structure:

telco-customer-churn/
 └── WA_Fn-UseC_-Telco-Customer-Churn.csv

▶️ Usage

Run the notebook
Open customer_churn_prediction_using_machine_learning.ipynb in Jupyter or VS Code and execute the cells.

Train and evaluate models

Decision Tree, Random Forest, and XGBoost are trained.

Performance metrics are displayed for comparison.

Export trained model
The best-performing model is saved as:

customer_churn_model.pkl


You can load it in Python:

import pickle
model = pickle.load(open("customer_churn_model.pkl", "rb"))


Make predictions

sample = [[...]]  # input features
prediction = model.predict(sample)
print("Churn" if prediction[0] == 1 else "Not Churn")

📊 Example Results
Model	Accuracy	ROC-AUC
Decision Tree	~78%	~0.75
Random Forest	~80–82%	~0.83
XGBoost	~84–86%	~0.85

(Exact values depend on train-test split & preprocessing)

🔮 Future Work

Hyperparameter tuning with GridSearchCV / Optuna.

Feature importance visualization.

Deployment with Flask/FastAPI.

Model explainability with SHAP.

📜 License

MIT License
