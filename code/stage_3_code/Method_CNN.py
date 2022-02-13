'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD





from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN(method, nn.Module):
    data = None

    def __init__(self, mName, mDescription, _max_epoch, _learning_rate, device, chan, f, g):
        self.device=device
        self.max_epoch = _max_epoch
        self.learning_rate = _learning_rate
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.f=f
        self.g=g


    def forward(self, x):
       # print(x)
        x=self.f(x)
        #print(x)
        x=x.contiguous().view(x.size(0), -1)
        x=self.g(x)
        return x


    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        loss_list=[]
        x=torch.FloatTensor(np.array(X)).to(self.device)
        y_true=torch.LongTensor(np.array(y)).to(self.device)
        #y_true=torch.nn.functional.one_hot(y_true)
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            
            y_pred = self.forward(x)
            # convert y to torch.tensor as well
            #print(y_pred)
            #print(y_true)
            # calculate the training loss

            train_loss = loss_function(y_pred, y_true)
            
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            optimizer.zero_grad()
            
            train_loss.backward()
            
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()
            
            if epoch%100 == 0:
                loss_list.append(float(train_loss.item()))
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
        return(loss_list)
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.device))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        #print('method running...')
        #print('--start training...')
        loss_list=self.train(self.data['train']['X'], self.data['train']['y'])
        #print('--start testing...')
        pred_y = self.test(self.data['test']['X']).cpu()
        #print(pred_y)
        #print("pred: ", pred_y, " true: ", self.data['test']['y'])
        return {'result':{'pred_y': pred_y, 'true_y': self.data['test']['y']},'loss':loss_list}
            