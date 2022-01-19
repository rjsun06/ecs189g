'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

    def evaluate(self, average):
        result=self.data
        print('testing',self.evaluate_name)
        print("overall acc: ", accuracy_score(result['true_y'], result['pred_y']))
        print(average, "pre: ", precision_score(result['true_y'], result['pred_y'], average=average))
        print(average, "recal: ", recall_score(result['true_y'], result['pred_y'], average=average))
        print(average, "f1: ", f1_score(result['true_y'], result['pred_y'], average=average))