import pickle

# Wrapper function for constructing paths leading to dataset files
def construct_dataset_paths(module_name, config):
    module_config = extract_module_config(config, module_name)

    return config.datasets_root + module_config['dataset_train'], config.datasets_root + module_config['dataset_test']

def extract_module_config(config, module_name):
    for module in config.modules_config:
        if module['module_name'] == module_name:
            return module

    log('Failed to extract module config from config file.')

def load_pickle_data(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)

    return data

# Wrapper - adding functionality later (writing to file, formatting)
def log(msg):
    print(msg)