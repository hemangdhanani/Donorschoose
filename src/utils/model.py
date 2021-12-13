from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import math
import pandas as pd

def create_model(X_tr, y_train):
    nb = MultinomialNB(class_prior=[0.5,0.5])

    parameters = {'alpha':[0.00001, 0.0001,0.001, 0.01, 0.1,0.5,0.8, 1, 10,  100,  1000]}

    clf = GridSearchCV(nb, parameters, cv= 10, scoring='roc_auc',return_train_score=True,verbose=2)

    clf.fit(X_tr, y_train)
    df = pd.DataFrame(clf.cv_results_)
    # df.to_excel("outputGCV.xlsx")
    bestparameter = clf.best_params_
    train_auc = clf.cv_results_['mean_train_score']
    train_auc_std= clf.cv_results_['std_train_score']
    cv_auc = clf.cv_results_['mean_test_score']
    cv_auc_std= clf.cv_results_['std_test_score']

    nb_bow = MultinomialNB(alpha = 0.001,class_prior=[0.5,0.5])
    nb_bow.fit(X_tr, y_train)

    
    # print("="*30)
    # print(f"Best parameter value is : {bestparameter}")
    # print("="*30)
    # print(f"train_auc is : {train_auc}")
    # print("="*30)
    # print(f"train_auc_std is : {train_auc_std}")
    # print("="*30)
    # print(f"cv_auc is : {cv_auc}")
    # print("="*30)
    # print(f"cv_auc_std is : {cv_auc_std}")
    # print("="*30)



