from utils.common import read_config
import argparse
from utils.data_mgmt import get_data
from utils.data_mgmt import get_data_overview
from utils.data_mgmt import get_eda_results
from utils.data_mgmt import data_preprocessing
from utils.data_mgmt import data_vectorization_process
from models.naive_bayes import multinomial_naive_bayes
from models.logistic_regression import logistic_regression_model
from models.svm import svm_rbf_kernel_medel
from models.random_forest import random_forest_model

def training(config_path):
    config = read_config(config_path)
    testing_datasize = config['params']['testing_datasize'] #TO DO from config.yaml
    train_data, resource_data = get_data()
    get_data_overview(train_data, resource_data)
    get_eda_results(train_data, resource_data)
    train_data_clean = data_preprocessing(train_data, resource_data)
    X_tr_vec, X_test_vec, X_cv_vec, y_train, y_test, y_cv = data_vectorization_process(train_data_clean)    
    multinomial_naive_bayes(X_tr_vec, X_cv_vec, X_test_vec, y_train, y_cv, y_test)
    logistic_regression_model(X_tr_vec, X_cv_vec, X_test_vec, y_train, y_cv, y_test)
    svm_rbf_kernel_medel(X_tr_vec, X_cv_vec, X_test_vec, y_train, y_cv, y_test)
    random_forest_model(X_tr_vec, X_cv_vec, X_test_vec, y_train, y_cv, y_test)
    #model()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # args = parser.parse_args()
    args.add_argument("--config","-c", default="config.yaml")
    parse_args = args.parse_args()
    training(config_path=parse_args.config)