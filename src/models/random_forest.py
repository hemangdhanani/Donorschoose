from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import joblib

def random_forest_model(X_tr, X_cv_vec, X_test, y_train, y_cv, y_test):
    alpha = [100,200,500,1000,2000]
    max_depth = [5, 10]
    cv_log_error_array = []
    for i in alpha:
        for j in max_depth:
            print("for n_estimators =", i,"and max depth = ", j)
            clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42)
            clf.fit(X_tr, y_train)
            sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
            sig_clf.fit(X_tr, y_train)
            sig_clf_probs = sig_clf.predict_proba(X_cv_vec)
            cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))
            print("Log Loss :",log_loss(y_cv, sig_clf_probs))

    best_alpha = np.argmin(cv_log_error_array)
    clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42)
    clf.fit(X_tr, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_tr, y_train)
    joblib.dump(clf, 'random_forest_model.pkl')

    predict_y = sig_clf.predict_proba(X_tr)
    print('For values of best estimator = ', alpha[int(best_alpha/2)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_cv_vec)
    print('For values of best estimator = ', alpha[int(best_alpha/2)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_test)
    print('For values of best estimator = ', alpha[int(best_alpha/2)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    y_pred = sig_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print(f"accuracy score for linear svm: {accuracy}")




