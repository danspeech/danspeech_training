import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import torch
from deepspeech.train import finetune, train_new, continue_training
if torch.cuda.is_available():
    print("Success")
    print(torch.cuda.device_count())

if __name__ == '__main__':    
    #train_new(model_id=None, train_data_path='/scratch/s134843/danspeech/', validation_data_path='/scratch/s134843/danspeech/', cuda=True)
    #continue_training(model_id='danish_speaking_panda_finetuned_continued', 
    #                  train_data_path='/scratch/s134843/danspeech/',
    #                  validation_data_path='/scratch/s134843/danspeech/', 
    #                  stored_model = '/home/s123106/.danspeech/models/danish_speaking_panda_finetuned.pth',
    #                  cuda=True)

    #finetune(model_id='danish_speaking_panda_finetuned',
    #         train_data_path='/scratch/s134843/danspeech/',
    #         validation_data_path='/scratch/s134843/danspeech',
    #         stored_model='/home/s123106/.danspeech/models/danish_speaking_panda.pth',
    #         cuda=True)

