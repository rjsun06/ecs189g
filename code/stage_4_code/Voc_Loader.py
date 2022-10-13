'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Voc_Loader(dataset):
    #print([])
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    setType= None

    vocnum=0
    transferl={'!':1,':':2,',':3,'.':4,'\'s':5,'1':6,'...':7,'\"':8,'(':9,')':10}
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.setType=None

    def transfer(self,word):
        return self.vocnum+self.transferl[word]

    def add(self,res,word):
        #print(word)
        if word.isnumeric():
            res.append(self.transfer('1'))
        elif word == '':
            res.append(-1)

        elif word[0]=='\"':
            res.append(self.transfer('\"'))
            if len(word) > 1:
                self.add(res,word[1:])

        elif word[-1]=='\"':
            if len(word) > 1:
                self.add(res,word[:-1])
            res.append(self.transfer('\"'))

        elif word[0]=='(':
            res.append(self.transfer('('))
            if len(word) > 1:
                self.add(res,word[1:])

        elif word[-1]==')':
            if len(word) > 1:
                self.add(res,word[:-1])
            res.append(self.transfer(')'))
        elif word[-2:] == "'s":

            if len(word) > 2:
                self.add(res,word[:-2])
            res.append(self.transfer('\'s'))

        elif word[-1:] == "'":
            if len(word) > 1:
                self.add(res,word[:-1])
            res.append(self.transfer('\'s'))

        elif word[-1] == ':':
            if len(word) > 1:
                self.add(res,word[:-1])
            res.append(self.transfer(':'))

        elif word[-1] == '!':
            if len(word) > 1:
               self.add(res,word[:-1])
            res.append(self.transfer('!'))

        elif word[-3:]=='n\'t':
            if len(word) > 3:
                self.add(res,word[:-3])
            res=self.add(res,'not')

        elif word[-3:]=='...':
            if len(word) > 3:
                self.add(res,word[:-3])
            res.append(self.transfer('...'))

        elif word[-1]=='.':
            if len(word) > 1:
                self.add(res,word[:-1])
            res.append(self.transfer('.'))

        elif word[-1]==',':
            if len(word) > 1:
                self.add(res,word[:-1])
            res.append(self.transfer(','))

        else:
            if word!= '':
                tmp=word.lower()
                self.voc.setdefault(tmp,-2)
                if self.voc[tmp]!=-2:
                    res.append(self.voc[tmp])
               # print(res)

    def tok(self,raw):

        res=[]
        for word in str.split(str(raw, encoding = "utf8"),' '):
            self.add(res,word)

        return res

    def load(self):
        print('loading Voc...ff')

        vocpath=self.dataset_source_folder_path+self.dataset_source_file_name+'/'+ 'imdb.vocab'



        with open(vocpath, 'rb') as f:
            keys=f.readlines()
        self.vocnum=len(keys)
        #print(keys)
        for i,s in enumerate(keys):
            #print(str(s,encoding = "utf8").strip('\n'))
            keys[i]=(str(s,encoding = "utf8").strip('\n'))

        self.voc = dict(zip(keys, range(0, len(keys))))




        return self.voc