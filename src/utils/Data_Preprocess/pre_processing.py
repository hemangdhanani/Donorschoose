import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import re

def get_clean_category(train_data):
    categories = list(train_data['project_subject_categories'].values)
    cat_list = []
    for i in categories:
        temp = ''
        for j in i.split(','):
            if 'The' in j.split():
                j = j.replace('The', '')
            j = j.replace(' ', '')
            temp = temp + j.strip() + " "
            temp = temp.replace('&', '_')
        cat_list.append(temp.strip())

    train_data['clean_categories'] = cat_list
    return train_data

def get_clean_sub_category(train_data): 
    subject_subcat = list(train_data['project_subject_subcategories'].values)
    sub_cat_list = []
    for i in subject_subcat:
        temp = ''
        for j in i.split(','):
            if 'The' in j.split():
                j = j.replace('The', '')
            j = j.replace(' ', '')
            temp = temp+j.strip()+' '
            temp = temp.replace('&','_')
        sub_cat_list.append(temp.strip())

    train_data['clean_subcategories'] = sub_cat_list
    return train_data

def get_total_eassay(train_data):
    train_data["essay"] = train_data["project_essay_1"].map(str) +\
                        train_data["project_essay_2"].map(str) + \
                        train_data["project_essay_3"].map(str) + \
                        train_data["project_essay_4"].map(str)
    return train_data                    
    
def get_clean_project_grade_category(train_data):
    train_data['project_grade_category'] = train_data['project_grade_category'].str.replace(' ', '_')
    train_data['project_grade_category'] = train_data['project_grade_category'].str.replace('-','_')
    train_data['project_grade_category'] = train_data['project_grade_category'].str.lower()
    project_grade_cat = train_data['project_grade_category'].value_counts()
    print("=" * 30)
    print(f"Project Grade category counts are :")
    print(f"{project_grade_cat}")
    print("=" * 30)
    return train_data

def get_clean_project_subject_categories(train_data):
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.replace(' The ','')
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.replace(' ','')
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.replace('&','_')
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.replace(',','_')
    train_data['project_subject_categories'] = train_data['project_subject_categories'].str.lower()

    project_subject_cate = train_data['project_subject_categories'].value_counts()
    print("=" * 30)
    print(f"Project subject categories counts are :")
    print(f"{project_subject_cate}")
    print("=" * 30)
    return train_data

def get_clean_teacher_prefix(train_data):
    train_data['teacher_prefix'] = train_data['teacher_prefix'].fillna('Mrs.')
    train_data['teacher_prefix'] = train_data['teacher_prefix'].str.replace('.','')
    train_data['teacher_prefix'] = train_data['teacher_prefix'].str.lower()
    train_data['teacher_prefix'].value_counts()

    teacher_prefix_cate = train_data['teacher_prefix'].value_counts()
    print("=" * 30)
    print(f"teacher_prefix categories counts are :")
    print(f"{teacher_prefix_cate}")
    print("=" * 30)
    return train_data

def get_clean_subject_sub_category(train_data):
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.replace(' The ','')
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.replace(' ','')
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.replace('&','_')
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.replace(',','_')
    train_data['project_subject_subcategories'] = train_data['project_subject_subcategories'].str.lower()
    return train_data 

def get_clean_school_state_data(train_data):
    train_data['school_state'] = train_data['school_state'].str.lower()
    train_data['school_state'].value_counts()
    return train_data

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_data):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text

def get_clean_project_title(train_data):
    preprocessed_titles = preprocess_text(train_data['project_title'].values)
    train_data['project_title'] = preprocessed_titles
    return train_data

def get_clean_essay(train_data):
    preprocessed_essays = preprocess_text(train_data['essay'].values)
    train_data["essay"] = preprocessed_essays
    return train_data

def get_normalized_std_price(train_data):
    # scaler_std = StandardScaler()
    # # scaler_std.fit(train_data['price'].values.reshape(-1, 1))
    # train_data['std_price'] = scaler_std.fit_transform(train_data['price'].values.reshape(-1, 1) )

    # scaler_norm = MinMaxScaler()
    # # scaler_norm.fit(train_data['price'].values.reshape(-1, 1))
    # train_data['nrm_price']=scaler_norm.fit_transform(train_data['price'].values.reshape(-1, 1))

    return train_data

def get_project_resource_summary(train_data):
    preprocessed_resource_summary = preprocess_text(train_data['project_resource_summary'].values)
    train_data['project_resource_summary'] = preprocessed_resource_summary
    return train_data

def drop_columns(train_data):
    train_data.drop(['Unnamed: 0'], axis=1, inplace=True)
    train_data.drop(['teacher_id'], axis=1, inplace=True)
    train_data.drop(['project_submitted_datetime'], axis=1, inplace=True)
    train_data.drop(['project_essay_1'], axis=1, inplace=True)
    train_data.drop(['project_essay_2'], axis=1, inplace=True)
    train_data.drop(['project_essay_3'], axis=1, inplace=True)
    train_data.drop(['project_essay_4'], axis=1, inplace=True)
    return train_data










