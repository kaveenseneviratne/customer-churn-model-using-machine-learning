# Customer Churn Prediction using Machine Learning

This project builds a **machine learning model to predict customer churn** using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
It applies preprocessing, class balancing, and multiple ML algorithms to identify customers likely to leave a telecom service provider.

---

## 📦 Features
- Automatic dataset download via **Kaggle API** (with `opendatasets`)
- Data preprocessing: missing values, categorical encoding, scaling
- **SMOTE** oversampling to address class imbalance
- Multiple models trained & compared:
  - Decision Tree
  - Random Forest
  - XGBoost
- Evaluation metrics: accuracy, confusion matrix, classification report (and optional ROC-AUC)
- Best model saved as a **pickle file** for reuse

---

## ⚙️ Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-churn-ml.git
cd customer-churn-ml
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

> Typical requirements (if you need a starting point):  
> `pandas numpy scikit-learn imbalanced-learn xgboost opendatasets matplotlib`

3. **Kaggle API setup**
- Go to [Kaggle → Account → Create New API Token](https://www.kaggle.com/account) to download `kaggle.json`.
- Place `kaggle.json` in:
  - **Linux/macOS**: `~/.kaggle/kaggle.json`
  - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
- Ensure the file permissions are restricted (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/macOS).

---

## 📂 Dataset

Download the dataset automatically with `opendatasets`:

```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
```

This will create:
```
telco-customer-churn/
 └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## ▶️ Usage

1. **Run the notebook**  
Open `customer_churn_prediction_using_machine_learning.ipynb` in Jupyter or VS Code and execute all cells in order.

2. **Train & evaluate models**  
The notebook trains Decision Tree, Random Forest, and XGBoost, applies SMOTE, and prints evaluation metrics for comparison.

3. **Export the trained model**  
The best-performing model is saved as:
```
customer_churn_model.pkl
```

4. **Load the model for inference**
```python
import pickle
model = pickle.load(open("customer_churn_model.pkl", "rb"))

# Example: replace `sample` with your preprocessed feature rows
sample = [[...]]  
prediction = model.predict(sample)
print("Churn" if prediction[0] == 1 else "Not Churn")
```

> ⚠️ Make sure your inference pipeline applies the **same preprocessing** (encoding/scaling) used during training.

---

## 📊 Example Results

| Model         | Accuracy | ROC-AUC |
|---------------|----------|---------|
| Decision Tree | ~78%     | ~0.75   |
| Random Forest | ~80–82%  | ~0.83   |
| XGBoost       | ~84–86%  | ~0.85   |

*Exact values depend on train/test split, preprocessing, and random seed.*

---

## 🔮 Future Work
- Hyperparameter tuning (GridSearchCV / Optuna)
- Feature importance and SHAP explainability
- Flask/FastAPI service for real-time predictions
- Model monitoring & drift detection

---

## 📝 License
MIT License

---

## 🙌 Acknowledgments
Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Thanks to the open-source community for `scikit-learn`, `imbalanced-learn`, `xgboost`, and `opendatasets`.
