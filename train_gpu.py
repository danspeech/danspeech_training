import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
import torch
from deepspeech.train import train_new
if torch.cuda.is_available():
    print("Success")
    print(torch.cuda.device_count())

if __name__ == '__main__':
    train_new(model_id=None, train_data_path='/scratch/s134843/danspeech/',
              validation_data_path='/scratch/s134843/danspeech/', cuda=True)
