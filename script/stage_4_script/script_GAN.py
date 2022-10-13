from code.stage_4_code.Dataset_Loader_gan import Dataset_Loader
from code.stage_4_code.Method_GAN import Method_GAN
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Mini_Batch_gan import Setting_Mini_Batch
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Voc_Loader import Voc_Loader

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
    i = 1
    f =1
    # ---- objection initialization setction ---------------
    files=      ['text_classification',   'text_generation', 'try']
    sizes=      [4,     1,   ]
    epoches=    [1,    1,         ]
    stages=     [1000,  100,            ]

    embedding_dim = 128
    hidden_dim = 250
    output_dim = 10
    sequence_length = 2714

    vocab_size=89527+11

    lrs=[10e-5,10e-6,10e-4]
    fs=[
        nn.Sequential(

        # Step 1: Embed our dimensions! vocab --> each word embedded into a vector of 100
        # output: [batch_size, sequence_length, embedding_dim]
        nn.Embedding(vocab_size, embedding_dim),

        # Step 2: LSTM it
        # we input embedding_dim, assume 1 item in batch, then we have the LSTM neuron
        # process one word (one embedding dim) at a time
        nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1),

        # Step 3: Squash the output
        # We will not include sigmoid activation since we are using BCEWithLogits(), which
        # already computes the sigmoid for us.

        ).to(device),

        nn.Sequential(

            # Step 1: Embed our dimensions! vocab --> each word embedded into a vector of 100
            # output: [batch_size, sequence_length, embedding_dim]
            nn.Embedding(vocab_size, embedding_dim),

            # Step 2: LSTM it
            # we input embedding_dim, assume 1 item in batch, then we have the LSTM neuron
            # process one word (one embedding dim) at a time
            nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1),

            # Step 3: Squash the output
            # We will not include sigmoid activation since we are using BCEWithLogits(), which
            # already computes the sigmoid for us.

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

            nn.Linear(int((600000/2400)*sequence_length), 64),

            nn.Linear(64, 11)
        ).to(device),

        nn.Sequential(
            nn.Linear(int((600000/2400)*sequence_length), 64),

            nn.ReLU(),
            nn.Linear(64, 1)
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


    epoch = epoches[i]
    f=fs[i]
    g=gs[i]
    addaxis=i==0

    torch.cuda.empty_cache()

   # vocload_obj=trainset_obj = Voc_Loader(file, '')
   # vocload_obj.dataset_source_folder_path = '../../data/stage_4_data/'
   # vocload_obj.dataset_source_file_name = file



    trainset_obj = Dataset_Loader(file, )
    trainset_obj.dataset_source_folder_path = '../../data/stage_4_data/'
    trainset_obj.dataset_source_file_name = file
    trainset_obj.setType='train'
#    trainset_obj.voc=vocload_obj

    trainset_obj.addaxis=addaxis

    testset_obj = Dataset_Loader(file, )
    testset_obj.dataset_source_folder_path = '../../data/stage_4_data/'
    testset_obj.dataset_source_file_name = file
    testset_obj.setType='test'
#    testset_obj.voc = vocload_obj
   # testset_obj.addaxis=addaxis

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Mini_Batch('k fold cross validation', '')
    # setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------
    # for learning_rate in [10e-5,10e-6,10e-7]:
    #    for epoch in [500,1000,2000]:
    #        for size in [100,500,1000]:


    n_layers=1
    loss_function=nn.CrossEntropyLoss
    optimizer=torch.optim.Adam
    # ---- running section ---------------------------------
    method_obj = Method_GAN(vocab_size, n_layers, embedding_dim, hidden_dim, output_dim, epoch, learning_rate, loss_function,
                 optimizer, device).to(device)
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

