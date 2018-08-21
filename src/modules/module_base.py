import other.util as util

# The base class for all the modules
class ModuleBase:

    def __init__(self, module_name, config):
        self.module_name = module_name
        self.config = config

        self.path_train, self.path_test = util.construct_dataset_paths(module_name, config)
        # self.dataset_train, self.dataset_test = util.load_data(self.path_train, self.path_test)

        # Accessing features and labels
        # X_train, y_train = train['features'], train['labels']
        # X_test, y_test = test['features'], test['labels']

    
    def outputModuleInit(self):
        util.log('Module \'' + self.module_name + '\' init successful.')
