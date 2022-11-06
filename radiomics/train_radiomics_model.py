import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

def sensitivity_score(y, pred):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y, pred)
    sensitivity = cm[0,0] / (cm[0,0]+cm[0,1])
    return sensitivity

def specificity_score(y, pred):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y, pred)
    specificity = cm[1,1] / (cm[1,1]+cm[1,0])
    return specificity

def main():
    traindataset = pd.read_csv("radiomics/train_radiomics.csv")
    testdataset = pd.read_csv("radiomics/test_radiomics.csv")

    X_train = traindataset.iloc[:,4:]
    X_test = testdataset.iloc[:,4:]

    y_train = traindataset['3']
    y_test = testdataset['3']

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)

    y_test_pred_lr = lr.predict(X_test_scaled)
    y_test_proba_lr = lr.predict_proba(X_test_scaled)[:,1]

    print("Accuracy", accuracy_score(y_test, y_test_pred_lr))
    print("Sensitivity", sensitivity_score(y_test, y_test_pred_lr))
    print("Specificity", specificity_score(y_test, y_test_pred_lr))
    print("F1 score", f1_score(y_test, y_test_pred_lr))
    print("AUC", roc_auc_score(y_test, y_test_proba_lr))

if __name__ == "__main__":
    main()
