import pandas as pd
from utils.EDA.data_eda import ylabeloverview
from utils.EDA.data_eda import state_wise_acceptance_eda
from utils.EDA.data_eda import teacher_prefix_eda
from utils.EDA.data_eda import project_grade_eda
from utils.EDA.data_eda import project_title_eda
# from utils.EDA.data_eda import project_eassay_eda
from utils.EDA.data_eda import previous_submitted_projects
from utils.Data_Preprocess.pre_processing import get_clean_category
from utils.Data_Preprocess.pre_processing import get_clean_sub_category
from utils.Data_Preprocess.pre_processing import get_total_eassay
from utils.Data_Preprocess.pre_processing import get_clean_project_grade_category
from utils.Data_Preprocess.pre_processing import get_clean_project_subject_categories
from utils.Data_Preprocess.pre_processing import get_clean_teacher_prefix
from utils.Data_Preprocess.pre_processing import get_clean_subject_sub_category
from utils.Data_Preprocess.pre_processing import get_clean_school_state_data
from utils.Data_Preprocess.pre_processing import get_clean_project_title
from utils.Data_Preprocess.pre_processing import get_clean_essay
from utils.Data_Preprocess.pre_processing import get_normalized_std_price
from utils.Data_Preprocess.pre_processing import get_project_resource_summary
from utils.Data_Preprocess.pre_processing import drop_columns


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
    # project_eassay_eda(train_data)
    previous_submitted_projects(train_data)

def data_preprocessing(train_data, resource_data):
    train_data_category = get_clean_category(train_data)
    train_data_sub_cat = get_clean_sub_category(train_data_category)
    train_data_eassay = get_total_eassay(train_data_sub_cat)
    train_data_clean_grade_cat = get_clean_project_grade_category(train_data_eassay)
    train_data_subject_cat = get_clean_project_subject_categories(train_data_clean_grade_cat)
    train_data_teacher_prefix = get_clean_teacher_prefix(train_data_subject_cat)
    train_data_sub_cat = get_clean_subject_sub_category(train_data_teacher_prefix)
    train_data_school_state = get_clean_school_state_data(train_data_sub_cat)
    train_data_project_title = get_clean_project_title(train_data_school_state)
    train_data_clean_essay = get_clean_essay(train_data_project_title)
    train_data_price_normalized = get_normalized_std_price(train_data_clean_essay)
    train_data_resource_summary = get_project_resource_summary(train_data_price_normalized)
    train_data_clean = drop_columns(train_data_resource_summary)
    # train_data_clean.to_csv("1.1_train_data_eassay.csv")

    

