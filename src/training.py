from utils.common import read_config
import argparse
from utils.data_mgmt import get_data


def training(config_path):
    config = read_config(config_path)
    testing_datasize = config['params']['testing_datasize']
    train_data, resource_data = get_data()
    train_data_null = train_data.isnull().sum()
    resource_data_null = resource_data.isnull().sum()
    print(f"null data for training set")
    print(train_data_null)
    print(f"null data for resource set")
    print(resource_data_null)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # args = parser.parse_args()
    args.add_argument("--config","-c", default="config.yaml")
    parse_args = args.parse_args()
    training(config_path=parse_args.config)