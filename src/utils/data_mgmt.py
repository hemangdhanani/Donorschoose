import pandas as pd

def get_data():
    train_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\Assignment_08\train_data.csv")
    resource_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\Assignment_08\resources.csv")
    return (train_data, resource_data)