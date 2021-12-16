import warnings
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from scipy.sparse import hstack


def drop_nans(data):
    data = data.dropna()
    return data

def train_test_split_data(data):
    y = data['project_is_approved'].values
    X = data.drop(['project_is_approved'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y,random_state=42)
       
    print(f"X_test len is 2 --> {len(X_test)} ")
    print("=="*50)
    return X_train, X_test, y_train, y_test

def train_test_cv_split_data(X_train, X_test, y_train, y_test):
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
    print(f"X_train len is 1 --> {len(X_train)}")
    print(f"X_test len is 2 --> {len(X_cv)} ")
    print("=="*50)
    return X_train, X_cv, y_train, y_cv

def school_state_ohe(X_train, X_test, X_cv):
    vectorizer_school_state = CountVectorizer()
    vectorizer_school_state.fit(X_train['school_state'].values)
    school_state_oho = vectorizer_school_state.transform(X_train['school_state'].values)
    school_state_oho_test = vectorizer_school_state.transform(X_test['school_state'].values)
    school_state_oho_cv = vectorizer_school_state.transform(X_cv['school_state'].values)
    return school_state_oho, school_state_oho_test, school_state_oho_cv

def teacher_prefix_ohe(X_train, X_test, X_cv):
    vectorizer_prefix = CountVectorizer()
    vectorizer_prefix.fit(X_train['teacher_prefix'].values)
    teacher_pre_oho = vectorizer_prefix.transform(X_train['teacher_prefix'].values)
    teacher_pre_oho_test = vectorizer_prefix.transform(X_test['teacher_prefix'].values)
    teacher_pre_oho_cv = vectorizer_prefix.transform(X_cv['teacher_prefix'].values)
    return teacher_pre_oho, teacher_pre_oho_test, teacher_pre_oho_cv

def project_grade_category_ohe(X_train, X_test, X_cv):
    vectorizer_grade  = CountVectorizer()
    vectorizer_grade.fit(X_train['project_grade_category'].values)
    project_grade_oho = vectorizer_grade.transform(X_train['project_grade_category'].values)
    project_grade_oho_test = vectorizer_grade.transform(X_test['project_grade_category'].values)
    project_grade_oho_cv = vectorizer_grade.transform(X_cv['project_grade_category'].values)
    return project_grade_oho, project_grade_oho_test, project_grade_oho_cv

def previous_submitted_projects_norm(X_train, X_test, X_cv):    
    normalizer = Normalizer()
    # normalizer.fit(X_train['teacher_number_of_previously_posted_projects'].values.reshape(1,-1))

    X_train_posted_project_norm = normalizer.fit_transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(1,-1))
    X_train_posted_project_norm_test = normalizer.fit_transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(1,-1))
    X_train_posted_project_norm_cv = normalizer.fit_transform(X_cv['teacher_number_of_previously_posted_projects'].values.reshape(1,-1))

    X_train_posted_project_norm = X_train_posted_project_norm.reshape(-1, 1)
    X_train_posted_project_norm_test = X_train_posted_project_norm_test.reshape(-1, 1)
    X_train_posted_project_norm_cv = X_train_posted_project_norm_cv.reshape(-1, 1)

    return X_train_posted_project_norm, X_train_posted_project_norm_test, X_train_posted_project_norm_cv

def clean_categories(X_train, X_test, X_cv):
    vectorizer_cat = CountVectorizer()
    vectorizer_cat.fit(X_train['clean_categories'].values)
    clean_categories_oho = vectorizer_cat.transform(X_train['clean_categories'].values)
    clean_categories_oho_test = vectorizer_cat.transform(X_test['clean_categories'].values)
    clean_categories_oho_cv = vectorizer_cat.transform(X_cv['clean_categories'].values)
    return clean_categories_oho, clean_categories_oho_test, clean_categories_oho_cv

def clean_sub_categories(X_train, X_test, X_cv):
    vectorizer_subcat = CountVectorizer()
    vectorizer_subcat.fit(X_train['clean_subcategories'].values)
    clean_subcategories_oho = vectorizer_subcat.transform(X_train['clean_subcategories'].values)
    clean_subcategories_oho_test = vectorizer_subcat.transform(X_test['clean_subcategories'].values)
    clean_subcategories_oho_cv = vectorizer_subcat.transform(X_cv['clean_subcategories'].values)
    return clean_subcategories_oho, clean_subcategories_oho_test, clean_subcategories_oho_cv

def price_normalized(X_train, X_test, X_cv):
    normalizer = Normalizer()
    # normalizer.fit(X_train['price'].values.reshape(1, -1))
    price_norm = normalizer.fit_transform(X_train['price'].values.reshape(1,-1))
    price_norm_test = normalizer.fit_transform(X_test['price'].values.reshape(1,-1))
    price_norm_cv = normalizer.fit_transform(X_cv['price'].values.reshape(1,-1))

    price_norm = price_norm.reshape(-1, 1)
    price_norm_test = price_norm_test.reshape(-1, 1)
    price_norm_cv = price_norm_cv.reshape(-1, 1)
    return price_norm, price_norm_test, price_norm_cv

def essay_ohe(X_train, X_test, X_cv):    
    vectorizer = TfidfVectorizer(min_df=10)
    vectorizer.fit(X_train['essay'].values)
    essay_tfidf = vectorizer.transform(X_train['essay'].values)
    essay_tfidf_test = vectorizer.transform(X_test['essay'].values)
    essay_tfidf_cv = vectorizer.transform(X_cv['essay'].values)
    return essay_tfidf, essay_tfidf_test, essay_tfidf_cv

def project_title_ohe(X_train, X_test, X_cv):
    vectorizer_bow_title = TfidfVectorizer(min_df=10)
    vectorizer_bow_title.fit(X_train['project_title'].values)

    project_title_tfidf = vectorizer_bow_title.transform(X_train['project_title'].values)
    project_title_tfidf_test = vectorizer_bow_title.transform(X_test['project_title'].values)
    project_title_tfidf_cv = vectorizer_bow_title.transform(X_cv['project_title'].values)

    return project_title_tfidf, project_title_tfidf_test, project_title_tfidf_cv

def merge_train_vectorization_columns(school_state_oho, teacher_pre_oho, project_grade_oho, X_train_posted_project_norm,
                                        clean_categories_oho, clean_subcategories_oho, price_norm, essay_tfidf, project_title_tfidf):
                                        
                                        X_tr = hstack((school_state_oho,teacher_pre_oho,project_grade_oho, X_train_posted_project_norm,clean_categories_oho, clean_subcategories_oho, price_norm, essay_tfidf, project_title_tfidf)).tocsr()

                                        return X_tr

def merge_test_vectorization_columns(school_state_oho_test, teacher_pre_oho_test, project_grade_oho_test, X_train_posted_project_norm_test,
                                        clean_categories_oho_test, clean_subcategories_oho_test, price_norm_test, essay_tfidf_test, project_title_tfidf_test):

                                        X_test = hstack((school_state_oho_test,teacher_pre_oho_test ,project_grade_oho_test, X_train_posted_project_norm_test ,clean_categories_oho_test, clean_subcategories_oho_test, price_norm_test,essay_tfidf_test, project_title_tfidf_test)).tocsr()

                                        return X_test
                                        
def merge_cv_vectorization_columns(school_state_oho_cv, teacher_pre_oho_cv, project_grade_oho_cv, X_train_posted_project_norm_cv,
                                    clean_categories_oho_cv, clean_subcategories_oho_cv, price_norm_cv, essay_tfidf_cv, project_title_tfidf_cv):

                                    X_cv = hstack((school_state_oho_cv, teacher_pre_oho_cv, project_grade_oho_cv, X_train_posted_project_norm_cv,clean_categories_oho_cv, clean_subcategories_oho_cv, price_norm_cv, essay_tfidf_cv, project_title_tfidf_cv)).tocsr()

                                    return X_cv






