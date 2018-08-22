import os.path as path

import other.util as util
import other.pickler as p

# The base class for all the modules
class ModuleBase:

    def __init__(self, module_name, loader, config):
        self.module_name = module_name
        self.loader = loader
        self.config = config

        self.path_train, self.path_test = util.construct_dataset_paths(module_name, config)
        self.prepare_dataset()

        self.outputModuleInit()

    def prepare_dataset(self):
        pickle_train = path.join(self.path_train, "..", self.module_name + ".train.p")
        pickle_test = path.join(self.path_test, "..", self.module_name + ".test.p")

        if path.isfile(pickle_train):
            self.module_log('Loading training set from pickle file.')
            self.dataset_train = p.load(pickle_train)
        else:
            self.module_log('Training pickle file does not exist - loading from dataset.')
            data = self.loader.load(self.path_train)
            self.dataset_train = p.save(data, pickle_train)

        if path.isfile(pickle_test):
            self.module_log('Loading test set from pickle file.')
            self.dataset_test = p.load(pickle_test)
        else:
            self.module_log('Test pickle file does not exist - loading from dataset.')
            data = self.loader.load(self.path_test)
            self.dataset_test = p.save(data, pickle_test)
    
    def outputModuleInit(self):
        self.module_log('\033[92mModule init successful.\033[0m')
        # self.module_log("Training dataset size: ", self.dataset_train['features'].shape[0])
        # self.module_log("Testing dataset size: ", self.dataset_test['features'].shape[0])

    def module_log(self, msg):
        util.log('[' + self.module_name + ']: ' + msg)
