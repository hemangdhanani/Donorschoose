from utils.common import read_config
import argparse
from utils.data_mgmt import get_data
from utils.data_mgmt import get_data_overview
from utils.data_mgmt import get_eda_results


def training(config_path):
    config = read_config(config_path)
    testing_datasize = config['params']['testing_datasize'] #TO DO from config.yaml
    train_data, resource_data = get_data()
    get_data_overview(train_data, resource_data)
    get_eda_results(train_data, resource_data)
     
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # args = parser.parse_args()
    args.add_argument("--config","-c", default="config.yaml")
    parse_args = args.parse_args()
    training(config_path=parse_args.config)