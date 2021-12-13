import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from prettytable import PrettyTable

def stack_plot(data, xtick, col2='project_is_approved', col3='total'):
    ind = np.arange(data.shape[0])

    plt.figure(figsize=(10, 4))
    p1 = plt.bar(ind, data[col3].values)
    p2 = plt.bar(ind, data[col2].values)

    plt.ylabel('Projects')
    plt.title('% of projects aproved state wise')
    plt.xticks(ind, list(data[xtick].values))
    plt.legend((p1[0], p2[0]), ('total', 'accepted'))
    # plt.show()

def univariate_barplots(data, col1, col2='project_is_approved', top=False):
    # Count number of zeros in dataframe python: https://stackoverflow.com/a/51540521/4084039
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@
    temp = pd.DataFrame(data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index()

    # Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039
    temp['total'] = pd.DataFrame(data.groupby(col1)[col2].agg(total='count')).reset_index()['total']
    temp['Avg'] = pd.DataFrame(data.groupby(col1)[col2].agg(Avg='mean')).reset_index()['Avg']

    temp.sort_values(by=['total'], inplace=True, ascending=False)

    if top:
        temp = temp[0:top]

    stack_plot(temp, xtick=col1, col2=col2, col3='total')
    print(temp.head(5))
    print("=" * 50)
    print(temp.tail(5))
    print("==" * 50)

def ylabeloverview(train_data):
    y_label_val_count = train_data['project_is_approved'].value_counts()
    y_approve = y_label_val_count[1]
    y_not_approve = y_label_val_count[0]
    y_total = y_label_val_count[1] + y_label_val_count[0]

    # ---------------------------------------------2.1 percentage of approved and non-approved----------------------------
    print("=="*50)
    print(f"Approve projects {y_approve}  and its percentage is {(y_approve/y_total) * 100}")
    print("=="*50)
    print(f"Not approve projects {y_not_approve}  and its percentage is {(y_not_approve/y_total) * 100}")
    print("=="*50)

    # ---------------------------------------------2.2 graphical representation of approved and non-approved----------------
    y_plot = np.array([y_approve, y_not_approve])
    mylabels = ["Approved", "Not approved"]
    plt.pie(y_plot, labels = mylabels)
    plt.show()

def state_wise_acceptance_eda(train_data):
    statewise_approval = pd.DataFrame(train_data.groupby("school_state")["project_is_approved"].apply(np.mean)).reset_index()
    statewise_approval.sort_values(by=['project_is_approved'], inplace=True)

    print(f"Top state for approval are ")
    print(statewise_approval.tail(5))
    print("=="*50)
    print(f"Last state for approval are ")
    print(statewise_approval.head(5))
    print("=="*50)

    univariate_barplots(train_data, 'school_state', 'project_is_approved', False)

def teacher_prefix_eda(train_data):
    teacher_prefix = pd.DataFrame(train_data.groupby('teacher_prefix')['project_is_approved'].apply(np.mean)).reset_index()
    teacher_prefix.sort_values(by=['project_is_approved'], inplace=True)
    print("Prefix wise approval")
    print(f"{teacher_prefix}")
    print("=="*50)
    univariate_barplots(train_data, 'teacher_prefix', 'project_is_approved', False)

def project_grade_eda(train_data): 
    project_grade = pd.DataFrame(train_data.groupby('project_grade_category')['project_is_approved'].apply(np.mean)).reset_index()
    project_grade.sort_values(by=['project_is_approved'], inplace=True)
    print("Grade wise approvals are")
    print(f"{project_grade}")
    print("=="*50)
    univariate_barplots(train_data, 'project_grade_category', 'project_is_approved', False)    

def project_title_eda(train_data):
    word_count = train_data['project_title'].str.split().apply(len).value_counts()
    word_dict = dict(word_count)
    word_dict = dict(sorted(word_dict.items(), key=lambda kv: kv[1]))

    ind = np.arange(len(word_dict))
    plt.figure(figsize=(10, 4))
    p1 = plt.bar(ind, list(word_dict.values()))
    plt.ylabel('Number of projects')
    plt.title('Words for each title of the project')
    plt.xticks(ind, list(word_dict.keys()))
    plt.show()

    approved_word_count = train_data[train_data['project_is_approved']== 1]['project_title'].str.split().apply(len)
    approved_word_count = approved_word_count.values

    rejected_word_count = train_data[train_data['project_is_approved']==0]['project_title'].str.split().apply(len)
    rejected_word_count = rejected_word_count.values

    plt.boxplot([approved_word_count, rejected_word_count])
    plt.xticks([1,2],('Approved Projects','Rejected Projects'))
    plt.ylabel('Words in project title')
    plt.grid()
    plt.show()
    
# def project_eassay_eda(train_data):
    # train_data["essay"] = train_data["project_essay_1"].map(str) +\
    #                     train_data["project_essay_2"].map(str) + \
    #                     train_data["project_essay_3"].map(str) + \
    #                     train_data["project_essay_4"].map(str)

    # word_count = train_data['essay'].str.split().apply(len).value_counts()
    # word_dict = dict(word_count)
    # word_dict = dict(sorted(word_dict.items(), key=lambda kv: kv[1]))


    # ind = np.arange(len(word_dict))
    # plt.figure(figsize=(20,5))
    # p1 = plt.bar(ind, list(word_dict.values()))

    # plt.ylabel('Number of projects')
    # plt.xlabel('Number of words in each eassay')
    # plt.title('Words for each essay of the project')
    # plt.xticks(ind, list(word_dict.keys()))
    # plt.show()

    # sns.distplot(word_count.values)
    # plt.title('Words for each essay of the project')
    # plt.xlabel('Number of words in each eassay')
    # plt.show()

    # approved_word_count = train_data[train_data['project_is_approved']==1]['essay'].str.split().apply(len)
    # approved_word_count = approved_word_count.values

    # rejected_word_count = train_data[train_data['project_is_approved']==0]['essay'].str.split().apply(len)
    # rejected_word_count = rejected_word_count.values

    # plt.boxplot([approved_word_count, rejected_word_count])
    # plt.title('Words for each essay of the project')
    # plt.xticks([1,2],('Approved Projects','Rejected Projects'))
    # plt.ylabel('Words in project title')
    # plt.grid()
    # plt.show()


    # plt.figure(figsize=(10,3))
    # sns.distplot(approved_word_count, hist=False, label="Approved Projects")
    # sns.distplot(rejected_word_count, hist=False, label="Not Approved Projects")
    # plt.title('Words for each essay of the project')
    # plt.xlabel('Number of words in each eassay')
    # plt.legend()
    # plt.show()

def previous_submitted_projects(train_data):  
    univariate_barplots(train_data, 'teacher_number_of_previously_posted_projects', 'project_is_approved', top=50)

    approved_previously_posted_projects_count=train_data[train_data['project_is_approved']==1]['teacher_number_of_previously_posted_projects'].values
    rejected_previously_posted_projects_count=train_data[train_data['project_is_approved']==0]['teacher_number_of_previously_posted_projects'].values

    plt.figure(figsize=(10,3))
    sns.distplot(approved_previously_posted_projects_count, hist=False, label="Approved Projects")
    sns.distplot(rejected_previously_posted_projects_count, hist=False, label="Not Approved Projects")
    plt.title('Number of previously posted projects by teacher')
    plt.xlabel('Number of previously posted projects')
    plt.legend()
    # plt.show()

    # from prettytable import PrettyTable
    # x = PrettyTable()
    # x.field_names = ["Percentile", "Approved Projects", "Not Approved Projects"]

    # for i in range(0,101,5):
    #     x.add_row([i,np.round(np.percentile(approved_previously_posted_projects_count,i), 3), np.round(np.percentile(rejected_previously_posted_projects_count,i), 3)])
    # print(x)

 