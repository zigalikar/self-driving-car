import pickle
from datetime import datetime
import os.path as path

# Wrapper function for constructing paths leading to dataset files
def construct_dataset_paths(module_name, config):
    return config.datasets_root + module_name + '\\train', config.datasets_root + module_name + '\\test'

def construct_weights_path(module_name, config, epoch=''):
    if epoch != '':
        return path.join(config.weights_root, module_name, 'weights-' + str(epoch) + '.ckpt')

    return path.join(config.weights_root, module_name)

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
def log(msg = ''):
    if len(msg) > 0:
        with open('logs/main-log.txt', 'a') as f: # TODO: log path to config
            file_entry = '({0}) - {1}\n'.format(datetime.now(), msg)
            f.write(file_entry)

    print(msg)