import other.util as util

# The base class for all the modules
class ModuleBase:

    def __init__(self, module_name, config):
        self.module_name = module_name
        self.config = config

        self.path_train, self.path_test = util.construct_dataset_paths(module_name, config)

        print('Module \'' + module_name + '\' init successful.')