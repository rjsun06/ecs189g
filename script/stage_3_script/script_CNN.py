from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Mini_Batch import Setting_Mini_Batch
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import os
from torch import nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# ---- Multi-Layer Perceptron script ----

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

if 1:

    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------
    i = 2
    f = 2
    # ---- objection initialization setction ---------------
    files=      ['MNIST',   'ORL',  'CIFAR' ]
    sizes=      [20000,     360,    2048       ]
    epoches=    [1000,      500,   10          ]
    stages=     [4,         1,      200        ]
    chans=      [1,         3,      3           ]
    lrs=[10e-5,10e-5,10e-4]
    fs=[nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 8, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8,16,3),
            nn.ReLU(inplace=True),

        ).to(device),

        nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(3, 1, 1),
            nn.Conv2d(1, 8, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    ).to(device),

        nn.Sequential(

            nn.Conv2d(3, 8, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)

        ).to(device)
    ]

    gs=[nn.Sequential(
            nn.Linear(144, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        ).to(device),

        nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 41)
        ).to(device),

        nn.Sequential(
            nn.Linear(128, 32),
            nn.Linear(32, 10),



        ).to(device)
        ]



    learning_rate = lrs[i]
    file=files[f]
    size=sizes[i]
    stage=stages[i]
    chan=chans[i]

    epoch = epoches[i]
    f=fs[i]
    g=gs[i]
    addaxis=i==0

    torch.cuda.empty_cache()
    trainset_obj = Dataset_Loader(file, '')
    trainset_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    trainset_obj.dataset_source_file_name = file
    trainset_obj.setType='train'
    trainset_obj.addaxis=addaxis

    testset_obj = Dataset_Loader(file, '')
    testset_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    testset_obj.dataset_source_file_name = file
    testset_obj.setType='test'
    testset_obj.addaxis=addaxis

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Mini_Batch('k fold cross validation', '')
    # setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------
    # for learning_rate in [10e-5,10e-6,10e-7]:
    #    for epoch in [500,1000,2000]:
    #        for size in [100,500,1000]:



    # ---- running section ---------------------------------
    method_obj = Method_CNN('cnn', '', epoch, learning_rate, device, chan,f,g).to(device)
    print('************ Start ************')
    setting_obj.prepare(trainset_obj, testset_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate(stage,size)
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ final evaluation ************')
    performance = setting_obj.do_evaluate()
    print('final performance: ' + str(performance))
    print('************ Finish ************')

    # ------------------------------------------------------

