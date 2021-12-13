import pandas as pd
from utils.EDA.data_eda import ylabeloverview
from utils.EDA.data_eda import state_wise_acceptance_eda
from utils.EDA.data_eda import teacher_prefix_eda
from utils.EDA.data_eda import project_grade_eda
from utils.EDA.data_eda import project_title_eda
from utils.EDA.data_eda import project_eassay_eda
from utils.EDA.data_eda import previous_submitted_projects

def get_data():
    train_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\Assignment_08\train_data.csv")
    resource_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\Assignment_08\resources.csv")
    return (train_data, resource_data)

def get_data_overview(train_data, resource_data):
    print("=="*30)
    print(f"train_data column names are: {train_data.columns}")
    print("=="*30)
    print(f"resource_data column names are: {resource_data.columns}")
    train_data_null = train_data.isnull().sum()
    resource_data_null = resource_data.isnull().sum()
    print("=="*30)
    print(f"null data for training set")
    print(train_data_null)
    print("=="*30)
    print(f"null data for resource set")
    print(resource_data_null)

def get_eda_results(train_data, resource_data):
    ylabeloverview(train_data)
    state_wise_acceptance_eda(train_data)
    teacher_prefix_eda(train_data)
    project_grade_eda(train_data)
    project_title_eda(train_data)
    project_eassay_eda(train_data)
    previous_submitted_projects(train_data)


    

