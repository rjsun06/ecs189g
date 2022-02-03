from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'
#---- Multi-Layer Perceptron script ----

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    trainset_obj = Dataset_Loader('stage2train', '')
    trainset_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    trainset_obj.dataset_source_file_name = 'train.csv'

    testset_obj = Dataset_Loader('stage2test', '')
    testset_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    testset_obj.dataset_source_file_name = 'test.csv'



    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------
    #for learning_rate in [10e-5,10e-6,10e-7]:
    #    for epoch in [500,1000,2000]:
    #        for size in [100,500,1000]:
    learning_rate=10e-5
    epoch=8000
    stage=1
    # ---- running section ---------------------------------
    method_obj = Method_MLP('multi-layer perceptron', '', epoch, learning_rate,device).to(device)
    print('************ Start ************')
    setting_obj.prepare(trainset_obj, testset_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate(stage)
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ final evaluation ************')
    performance = setting_obj.do_evaluate()
    print('final performance: ' + str(performance))
    print('************ Finish ************')

    # ------------------------------------------------------
    

    