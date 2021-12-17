from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,log_loss
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from .model_utils import predict_and_plot_confusion_matrix
from .model_utils import plot_confusion_matrix

def multinomial_naive_bayes(X_tr, X_cv_vec, X_test, y_train, y_cv, y_test):
    alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]
    cv_log_error_array = []
    for i in alpha:
        print("for alpha =", i)
        clf = MultinomialNB(alpha=i)
        clf.fit(X_tr, y_train)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(X_tr, y_train)
        sig_clf_probs = sig_clf.predict_proba(X_cv_vec)
        cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))
        # to avoid rounding error while multiplying probabilites we use log-probability estimates
        print("Log Loss :",log_loss(y_cv, sig_clf_probs)) 

    fig, ax = plt.subplots()
    ax.plot(np.log10(alpha), cv_log_error_array,c='g')
    for i, txt in enumerate(np.round(cv_log_error_array,3)):
        ax.annotate((alpha[i],str(txt)), (np.log10(alpha[i]),cv_log_error_array[i]))
    plt.grid()
    plt.xticks(np.log10(alpha))
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    # plt.show()


    best_alpha = np.argmin(cv_log_error_array)
    clf = MultinomialNB(alpha=alpha[best_alpha])
    clf.fit(X_tr, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_tr, y_train)


    predict_y = sig_clf.predict_proba(X_tr)

    print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_cv_vec)
    print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
    predict_y = sig_clf.predict_proba(X_test)
    print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    clf = MultinomialNB(alpha=alpha[best_alpha])
    clf.fit(X_tr, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(X_tr, y_train)
    sig_clf_probs = sig_clf.predict_proba(X_cv_vec)
    y_test_score = sig_clf.predict(X_test)
    # to avoid rounding error while multiplying probabilites we use log-probability estimates
    print("Log Loss :",log_loss(y_cv, sig_clf_probs))
    print("Number of missclassified point :", np.count_nonzero((sig_clf.predict(X_cv_vec)- y_cv))/y_cv.shape[0])
    plot_confusion_matrix(y_cv, sig_clf.predict(X_cv_vec.toarray()))

    # y_pred = sig_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_score, normalize=True)
    # print(f"accuracy score is {accuracy}")
    return accuracy
