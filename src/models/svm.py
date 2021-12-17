from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import numpy as np

def svm_rbf_kernel_model(X_tr, X_cv_vec, X_test, y_train, y_cv, y_test):
    alpha = [10 ** x for x in range(-5, 3)]
    cv_log_error_array = []
    for i in alpha:
        print("for C =", i)
    #     clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')
        clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='hinge', random_state=42)
        clf.fit(X_tr, y_train)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X_tr, y_train)
        sig_clf_probs = sig_clf.predict_proba(X_cv_vec)
        cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))
        print("Log Loss :",log_loss(y_cv, sig_clf_probs)) 

    fig, ax = plt.subplots()
    ax.plot(alpha, cv_log_error_array,c='g')
    for i, txt in enumerate(np.round(cv_log_error_array,3)):
        ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))
    plt.grid()
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    plt.show()


    best_alpha = np.argmin(cv_log_error_array)
    # clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')
    clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)
    clf.fit(X_tr, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_tr, y_train)

    predict_y = sig_clf.predict_proba(X_tr)
    print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_cv_vec)
    print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_test)
    print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    y_pred = sig_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print(f"accuracy score for linear svm: {accuracy}")

# def svm_rbf_kernel_medel(X_tr_vec, X_test_vec, y_train, y_test):
#     alpha = [10 ** x for x in range(-1, 2)]
#     # parameters = {'kernel':['rbf'], 'C':alpha}
#     param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
#     # svc_rbf_model = SVC()
#     grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
#     grid.fit(X_tr_vec,y_train)
#     print(grid.best_estimator_)
#     # clf = GridSearchCV(svc_rbf_model, param_grid, scoring='roc_auc',return_train_score=True,verbose=2)
#     # clf.fit(X_tr_vec, y_train)
#     # best_alpha2 = clf.best_params_['alpha']
#     # best_params_ = clf.best_params_
#     # print("="*30)
#     # print(f"best_params for SVM model--- : {best_params_}")
#     # print("="*30)    
