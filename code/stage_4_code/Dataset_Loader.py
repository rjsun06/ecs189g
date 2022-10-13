'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import csv
import pickle
import numpy as np
import os
import itertools

def zip_longest(ll,w):
  res=[[0 for i in range(len(ll))] for i in range(w)]

  for i in range(len(ll)):
      for j in range(len(ll[i])):
          res[j][i]=ll[i][j]

  return res

def duch(ll,w):
  res=[[0 for i in range(w)] for i in range(len(ll))]

  for i in range(len(ll)):
      for j in range(len(ll[i])):
          res[i][j]=ll[i][j]

  return res

class Dataset_Loader(dataset):
    #print([])
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    voc=None
    setType= None
    addaxis=False
    vocnum=0
    transferl={'!':1,':':2,',':3,'.':4,'\'s':5,'1':6,'...':7,'\"':8,'(':9,')':10}
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.setType=None



    def load(self):
        size=13500
        #size=5
        print('loading data...ff')



        path = self.dataset_source_folder_path+self.dataset_source_file_name+'/'+self.setType
        print(path)
        maxlen=2714
        X=[]
        y=[]
        for group in ['pos','neg']:
            newX=[[0]] *size
            newy=[0] * size
            for root,dirs,files in os.walk(os.path.join(path,group)):

                for file in files:
                    #print('1')
                    print(file)
                    with open(os.path.join(root,file),'rb') as f:
                        fn=str.split(str.split (file,'.')[0],'_')
                        #print(fn)
                        raw=f.readline()
                        pos=int(fn[0])
                        tmp=self.voc.tok(raw)
                        #if(len(tmp)>maxlen):maxlen=len(tmp)
                        newX[pos]=(tmp)
                        newy[pos]=(int(fn[1]))

            X=X+newX
            y=y+newy
        #print(maxlen)
        print(y)

        X = list(duch(X, maxlen))

        X=np.array(X,dtype=int)
        y=np.array(y,dtype=int)

        #print(X)




        return {'X': X, 'y': y}