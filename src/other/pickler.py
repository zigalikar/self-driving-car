import pickle

import other.util as util

# Saves dataset to a pickle file and returns the dataset as an object
def save(data, output):
    with open(output, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    util.log('Created a pickle file \'' + output + '\'.')

# Loading data from a pickle file
def load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)

    return data