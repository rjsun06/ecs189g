'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import csv
import pickle
import numpy as np

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    setType= None
    addaxis=False
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.setType=None
    def load(self):
        print('loading data...')
        X =[]
        y =[]
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            data = pickle.load(f)
        for pair in data[self.setType]:
                X.append(pair['image'])
                y.append(pair['label'])
        X=np.array(X)
        if self.addaxis:
            X=X[:,:,:, np.newaxis]
        #print(X)
        X=X.transpose((0, 3, 1, 2))
        y=np.array(y)
        #print(X)

        return {'X': X, 'y': y}