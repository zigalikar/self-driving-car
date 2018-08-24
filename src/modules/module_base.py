import os.path as path

import numpy as np

import other.util as util
import other.pickler as p

# The base class for all the modules
class ModuleBase:

    def __init__(self, module_name, loader, config):
        self.module_name = module_name
        self.loader = loader
        self.config = config
        self.module_config = util.extract_module_config(config, module_name)

        self.path_train, self.path_test = util.construct_dataset_paths(module_name, config)
        self.prepare_dataset() # TODO: check if weights exist and attempt to load them

        self.outputModuleInit()

    # Prepares the dataset for usage
    def prepare_dataset(self):
        pickle_train, pickle_test = util.construct_pickle_paths(self.module_name, self.config)
        self.dataset_train = self.load_dataset(pickle_train, self.path_train)
        self.dataset_test = self.load_dataset(pickle_test, self.path_test)
    
    # Loads a dataset from a pickled file or a dataset if pickle doesn't exist
    def load_dataset(self, pickle_path, dataset_path):
        if path.isfile(pickle_path):
            self.module_log('Loading dataset from pickle file: \'' + pickle_path + '\'')
            return p.load(pickle_path)
        else:
            self.module_log('Dataset pickle file does not exist - loading from dataset: \'' + dataset_path + '\'')
            data = self.loader.load(dataset_path)
            p.save(data, pickle_path)
            return data
        
    # Outputs an initialization message
    def outputModuleInit(self):
        self.module_log('\033[92m================== Module init successful ==================\033[0m')
        self.module_log('Training dataset size: ' + str(self.dataset_train['features'].shape[0]))
        self.module_log('Testing dataset size: ' + str(self.dataset_test['features'].shape[0]))
        self.module_log('Classifying into ' + str(np.unique(self.dataset_train['labels']).shape[0]) + ' classes.')

    # Wrapper - logs a message with the appropriate module ticker
    def module_log(self, msg):
        util.log('[' + self.module_name + ']: ' + msg)
