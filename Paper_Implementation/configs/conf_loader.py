import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# get configuration

# path of configuration file
config_path = os.path.dirname(os.path.abspath(__file__)) + '/config.yaml'

# load configuration file
conf = yaml.load(open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

dirichlet_alpha = conf['dirichlet']['alpha']

num_classes = conf['dataset']['num_classes']
dataset_name = conf['dataset']['name']
dataset_path = conf['dataset']['path']
batch_size = conf['dataset']['batch_size']

num_clients = conf['client']['num_clients']
client_ids = range(num_clients)
