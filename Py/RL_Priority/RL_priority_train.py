

import pickle
import kneed
import os
# from kneed import KneeLocator


# Load Pickled Data
print(os.getcwd())
filename = './RL_data/RLP_data_2020-08-06_09-53-56'
infile = open(filename, 'rb')
data_dict = pickle.load(infile)
infile.close()

a = 1

F = data_dict.get('F')
feat_record = data_dict.get('feat_record')
feature_names = data_dict.get('feature_names')


