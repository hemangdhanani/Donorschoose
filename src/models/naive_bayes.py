from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
def multinomial_naive_bayes(X_tr_vec, X_test_vec, y_train, y_test):
    nb = MultinomialNB(class_prior=[0.5,0.5])
    parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}

    clf = GridSearchCV(nb, parameters, cv= 10, scoring='roc_auc',return_train_score=True,verbose=2)

    clf.fit(X_tr_vec, y_train)
    bestparameter = clf.best_params_
    train_auc = clf.cv_results_['mean_train_score']
    train_auc_std = clf.cv_results_['std_train_score']
    cv_auc = clf.cv_results_['mean_test_score']
    cv_auc_std = clf.cv_results_['std_test_score']
    best_alpha2 = clf.best_params_['alpha']
    print("="*30)
    print(f"Best parameter value is : {bestparameter}")
    print("="*30)
    print(f"train_auc is : {train_auc}")
    print("="*30)
    print(f"train_auc_std is : {train_auc_std}")
    print("="*30)
    print(f"cv_auc is : {cv_auc}")
    print("="*30)
    print(f"cv_auc_std is : {cv_auc_std}")
    print("="*30)
    print(f"best_alpha2 is : {best_alpha2}")
    print("="*30)

    nb_tfidf = MultinomialNB(alpha = best_alpha2,class_prior = [0.5,0.5])
    nb_tfidf.fit(X_tr_vec, y_train)
    y_pred = nb_tfidf.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    print(cm)
    print(f"accuracy score is {accuracy}")
