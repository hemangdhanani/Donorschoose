from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

def logistic_regression_model(X_tr_vec, X_test_vec, y_train, y_test):
    alpha = [10 ** x for x in range(-6, 3)]

    logistic_regression_model = SGDClassifier(class_weight='balanced', penalty='l2', loss='log', random_state=42)   
    clf = GridSearchCV(logistic_regression_model, {'alpha':alpha}, cv= 10, scoring='roc_auc',return_train_score=True,verbose=2)    
    clf.fit(X_tr_vec, y_train)
    best_alpha2 = clf.best_params_['alpha']
    print("="*30)
    print(f"best_alpha2 is : {best_alpha2}")
    print("="*30)

    logistic_regression_model = SGDClassifier(class_weight='balanced', alpha=best_alpha2,penalty='l2', loss='log', random_state=42)
    logistic_regression_model.fit(X_tr_vec, y_train)
    y_pred = logistic_regression_model.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print(cm)
    print(f"accuracy score for logistic regression : {accuracy}")   
    # cv_log_error_array = []
    # for i in alpha:
    #     print("for alpha =", i)
    #     clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)
    #     clf.fit(X_tr_vec, y_train)
    #     sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    #     sig_clf.fit(X_tr_vec, y_train)
    #     sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
    #     cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    #     # to avoid rounding error while multiplying probabilites we use log-probability estimates
    #     print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 

    # fig, ax = plt.subplots()
    # ax.plot(alpha, cv_log_error_array,c='g')
    # for i, txt in enumerate(np.round(cv_log_error_array,3)):
    #     ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
    # plt.grid()
    # plt.title("Cross Validation Error for each alpha")
    # plt.xlabel("Alpha i's")
    # plt.ylabel("Error measure")
    # plt.show()