import pickle

def load_data(path_train, path_test):
    with open(path_train, mode='rb') as f:
        train = pickle.load(f)
    
    with open(path_test, mode='rb') as f:
        test = pickle.load(f)

    return train, test