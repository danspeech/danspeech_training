# DanSpeech Training add-on
## Repository structure
DanSpeech training is an add-on repository to the DanSpeech package. It was developed to help development cycles.

DanSpeech training supports three train wrapper functions all found in the deepspeech.train file. 

1. train_new(), trains a new model from strach
2. finetune(), takes a previosuly trained model and finetunes on new data
3. continue_train(), takes a previously trained model which was stopped during training and continues training.

Furthermore DanSpeech training provides a test script found in the deepspeech.test file. The test evaluates a trained model on a held-out test set and returns WER and CER estimates.

All training and testing requires a specific file structure.

1. All audio files associated with a particular task must be placed inside a folder (e.g. /home/usr/project_folder/train/ for training associated files).
2. Each task associated folder must include exactly one .csv file, each row in the .csv file is expected to point to exactly one audio file and contain all relevant information associated with the file it points to (e.g. /home/usr/project_folder/train/overview.csv).

## And example of a csv file could be:

row 0: file, transcription, gender, age

row 1: filename.wav, pandaer er et fantastisk dyr jeg ville ønske de kunne snakke dansk, mand, 34

row 2: filenmae1.wav, koalaer er søde jeg ville ønske de kunne snakke dansk, kvinde, 22

...

row n: filenamen.wav, pindsvin er flotte jeg ville ønske de ikke stak så meget, mand, 88

## Example use-cases:
training with a CPU: 
```python
from deepspeech.train import train_new

if __name__ == '__main__':
    train_new(model_id=None, train_data_path='/scratch/s134843/danspeech/', validation_data_path='/scratch/s134843/danspeech/')
```

training with a GPU:
```python
import os
import sys
import torch
from deepspeech.train import finetune, train_new, continue_training
if torch.cuda.is_available():
    print("Success")
    print(torch.cuda.device_count())

if __name__ == '__main__':
    # -- example code for training a new model
    train_new(model_id=None, train_data_path='/scratch/s134843/danspeech/', validation_data_path='/scratch/s134843/danspeech/', cuda=True)

    # -- example code for continuation of a new model
    continue_training(model_id='danish_speaking_panda_finetuned_continued',
                       train_data_path='/scratch/s134843/danspeech/',
                       validation_data_path='/scratch/s134843/danspeech/',
                       stored_model = '/home/s123106/.danspeech/models/danish_speaking_panda_finetuned.pth',
                       cuda=True)

    # -- example code of finetuning a model
    finetune(model_id='danish_speaking_panda_finetuned',
              train_data_path='/scratch/s134843/danspeech/',
              validation_data_path='/scratch/s134843/danspeech',
              stored_model='/home/s123106/.danspeech/models/danish_speaking_panda.pth',
              cuda=True)
```
## Authors and acknowledgment
Main authors: 
* Martin Carsten Nielsen  ([mcnielsen4270@gmail.com](mcnielsen4270@gmail.com))
* Rasmus Arpe Fogh Jensen ([rasmus.arpe@gmail.com](rasmus.arpe@gmail.com))

This project is supported by Innovation Foundation Denmark through the projects DABAI and ATEL

Other acknowledgements:

* We've trained the models based on the code from [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
* The audio handling and recognizing flow is based on [https://github.com/Uberi/speech_recognition](https://github.com/Uberi/speech_recognition).
* Handling of the pretrained models is based on [keras](https://github.com/keras-team/keras).
* We've trained all models with the aid of DTU using data from Sprakbanken ([NST](https://www.nb.no/sprakbanken/show?serial=oai%3Anb.no%3Asbr-19&lang=en)).
