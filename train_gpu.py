import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import torch
from deepspeech.train import finetune
if torch.cuda.is_available():
    print("Success")
    print(torch.cuda.device_count())

if __name__ == '__main__':
    finetune(model_id='danish_speaking_panda_finetuned', train_data_path='/scratch/s134843/danspeech/',
            validation_data_path='/scratch/s134843/danspeech/', epochs=20, cuda=True)
