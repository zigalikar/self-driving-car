import os.path as path

import other.util as util

# The base class for all the modules
class ModuleBase:

    def __init__(self, module_name, config):
        self.module_name = module_name
        self.config = config

        self.path_train, self.path_test = util.construct_dataset_paths(module_name, config)
        self.prepare_dataset()

        # Accessing features and labels
        # X_train, y_train = train['features'], train['labels']
        # X_test, y_test = test['features'], test['labels']

    def prepare_dataset(self):
        pickle_train = path.join(self.path_train, "..", self.module_name + ".train.p")
        pickle_test = path.join(self.path_test, "..", self.module_name + ".test.p")

        if path.isfile(pickle_train):
            self.module_log('Loading training set from pickle file.')
        else:
            self.module_log('Training pickle file does not exist - loading from images.')

        if path.isfile(pickle_test):
            self.module_log('Loading test set from pickle file.')
        else:
            self.module_log('Test pickle file does not exist - loading from images.')

        #self.dataset_train, self.dataset_test = util.load_data(self.path_train, self.path_test)
    
    # def outputModuleInit(self):
    #     util.log('Module \'' + self.module_name + '\' init successful.')
    #     util.log("Training dataset size: ", self.dataset_train['features'].shape[0])
    #     util.log("Testing dataset size: ", self.dataset_test['features'].shape[0])

    def module_log(self, msg):
        util.log('[' + self.module_name + ']: ' + msg)
