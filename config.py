import yaml

def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

config_file = 'config.yaml'
config = read_config(config_file)