'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import csv

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                elements=[int(i) for i in row]
                X.append(elements[1:])
                y.append(elements[0])
                #print(row[1:4])
                #print(row[0])
        f.close()
        return {'X': X, 'y': y}