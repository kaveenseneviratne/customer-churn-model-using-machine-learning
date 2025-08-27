# customer-churn-model-using-machine-learning
This is a customer churn prediction model built using machine learning (decision tree, random forest and XGBoost models)

# Customer Churn Prediction using Machine Learning  

This project builds a **machine learning model to predict customer churn** using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
It applies preprocessing, class balancing, and multiple ML algorithms to identify customers likely to leave a telecom service provider.  

---

##  Features  
- Automatic dataset download via Kaggle API.  
- Data preprocessing: handling missing values, encoding categorical features, scaling.  
- **SMOTE oversampling** to address class imbalance.  
- Multiple models tested:  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- Model evaluation with accuracy, confusion matrix, classification report.  
- Best model saved as a **pickle file** for reuse.  

---

##  Setup  

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/customer-churn-ml.git
cd customer-churn-ml

##  Setup

2. **Download the dataset** 
import opendatasets as od
od.download("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
