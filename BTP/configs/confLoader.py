import path
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# get configuration

# path of configuration file
config_path = os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'

# load configuration file
conf = yaml.load(open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

dataset_name = conf['data']['name']
dataset_root_path = conf['data']['root_path']
n_classes = conf['data']['n_classes']

n_clients = conf['metadata']['n_clients']
cuda = conf['metadata']['cuda']
global_epochs = conf['metadata']['global_epochs']
model_name = conf['metadata']['model']

client_train_batch_size = conf['client']['train_batch_size']
client_test_batch_size = conf['client']['test_batch_size']
client_lr = conf['client']['lr']
client_epochs = conf['client']['epochs']
client_data_distribution_type = conf['client']['data_distribution']
client_least_data = conf['client']['least_data']

global_batch_size = conf['global']['batch_size']
global_lr = conf['global']['lr']
random_select_ratio = conf['global']['random_select_ratio']
