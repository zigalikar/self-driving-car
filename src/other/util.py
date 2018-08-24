import pickle

import os.path as path

# Wrapper function for constructing paths leading to dataset files
def construct_dataset_paths(module_name, config):
    return config.datasets_root + module_name + '\\train', config.datasets_root + module_name + '\\test'

def construct_weights_path(module_name, epoch, config):
    return path.join(config.weights_root, module_name, '-', epoch, '.ckpt')

def construct_pickle_paths(module_name, config):
    path_train, path_test = construct_dataset_paths(module_name, config)
    return path.join(path_train, '..', module_name + '.train.p'), path.join(path_test, '..', module_name + '.test.p')

# Extracts module config from passed global config
def extract_module_config(config, module_name):
    for module in config.modules_config:
        if module['module_name'] == module_name:
            return module

    log('Failed to extract module config from config file.')

# Wrapper - adding functionality later (writing to file, formatting)
def log(msg):
    print(msg)