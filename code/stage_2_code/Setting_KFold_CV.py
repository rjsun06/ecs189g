'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as pl
import torch
class Setting_KFold_CV(setting):
    train   =None
    test    =None
    method  =None
    result  =None
    evaluate=None
    test_data = None

    def load_run_save_evaluate(self,stage):
        # load dataset
        self.train_data = self.train.load()
        self.test_data=self.test.load()
        train_data = self.train_data
        test_data=self.test_data
        
        score_list = []
        loss_list=[]
        for i in range(stage):
            test_index  =   np.random.randint(10000,size=3)
           # train_index =   np.random.randint(60000,size=_size)

            X_train, X_test = np.array(train_data['X']), np.array(test_data['X'])[test_index]
            y_train, y_test = np.array(train_data['y']), np.array(test_data['y'])[test_index]

            # run MethodModule
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
            
            learned_result = self.method.run()

            self.evaluate.data = learned_result['result']
            score_list.append(float(self.evaluate.evaluate()))

            loss_list+=float(learned_result['loss'])
        pl.plot(range(0,len(loss_list)*100,100),loss_list,label='loss', color='purple')
        pl.show()
        return np.mean(score_list), np.std(score_list)

    def do_evaluate(self):
        train_data =self.train_data
        test_data = self.test_data
        #test_index = np.random.randint(10000, size=size)

        pred_y = self.method.test(np.array(train_data['X'])).cpu()
        true_y = np.array(train_data['y'])
        self.evaluate.data = {'pred_y': pred_y, 'true_y': true_y}
        self.result.data = self.evaluate.data
        self.result.fold_count = 98
        self.result.save()

        pred_y=self.method.test(np.array(test_data['X'])).cpu()
        true_y = np.array(test_data['y'])
        self.evaluate.data ={'pred_y': pred_y, 'true_y': true_y}
        self.result.data=self.evaluate.data
        self.result.fold_count=99
        self.result.save()


        return self.evaluate.evaluate()

    def __init__(self, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription

    def prepare(self, _train, _test, sMethod, sResult, sEvaluate):
        self.train=_train
        self.test=_test
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print('trainset:', self.train.dataset_name, 'testset:', self.test.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name, ', evaluation:',
              self.evaluate.evaluate_name)