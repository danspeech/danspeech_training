# DanSpeech Training add-on
## Repository structure
DanSpeech training is an add-on repository to the DanSpeech package. It was developed to help development cycles.

DanSpeech training supports modes: 

1. train new model from scratch - train_new.py
2. continue training an existing model - train_continue.py
3. finetune an existing model - finetune.py

Furthermore, to evaulate models use the evaluate.py script.

All training and testing requires a specific file structure.

1. All audio files associated with a particular task must be placed inside a folder (e.g. /home/usr/project_folder/train/ for training associated files).
2. Each task associated folder must include exactly one .csv file, each row in the .csv file is expected to point to exactly one audio file and contain all relevant information associated with the file it points to (e.g. /home/usr/project_folder/train/overview.csv).

## Installation
To run the training code, follow the steps below. We suggest using a virtual environment.

1. Install [danspeech](https://github.com/danspeech/danspeech)
2. Install wget through pip 
3. Install [python bindings for warp-ctc](https://github.com/SeanNaren/warp-ctc)

To use beam search decoding, you need [ctc-decode](https://github.com/parlance/ctcdecode) as well.

The training repo has quite a lot of dependencies, hence this repo includes the `env_train.yml` file with
versions for all dependencies in a working virtual conda environment. If you have trouble, look at the file for the 
specific versions or try to install a virtual conda environment with:

```conda env create -f env_train.yml ```

Python bindings for warp-ctc always needs to be installed manually.

For multi GPU you additionally need [apex](https://github.com/NVIDIA/apex)

## And example of a csv file could be:
```
file,trans
filename.wav,pandaer er et fantastisk dyr jeg ville ønske de kunne snakke dansk
filenmae1.wav,koalaer er søde jeg ville ønske de kunne snakke dansk
...
filenamen.wav,pindsvin er flotte jeg ville ønske de ikke stak så meget
```
Due to current implementation, audio files need to be placed in same directory as the csv file. 
## Example use-cases:
Training with a CPU is default. To train with GPU parse --use_gpu to script.

See the args for each of the scripts for more information towards what you can parse.

#### Evaluate an existing model
```commandline
python evaluate.py --model_path ~/.danspeech/models/TestModel.pth --test_dataset ~/data/nst/preprocessed_test --transcriptions_out_file test.csv --use_gpu --batch_size 128 --beam_width 20 --decoder beam --lm_path ~/.danspeech/lms/dsl_3gram.klm --alpha 0.7
```

#### Training a new model
```commandline
python train_new.py --model_id test_id --train_data_path ~/data/nst/preprocessed_test --validation_data_path ~/data/nst/preprocessed_test --save_dir test_results --use_gpu --epochs 40 --batch_size 16 --train_with_augmentations 
```

#### Continue training an existing model
```commandline
python train_continue.py --model_id test_id --train_data_path ~/data/nst/preprocessed_test --validation_data_path ~/data/nst/preprocessed_test --save_dir test_results --use_gpu --epochs 40 --batch_size 16 --train_with_augmentations --continue_from_path ~/home/models/dummy.pth
```

#### Finetuning
```commandline
python finetune.py --model_id test_id --train_data_path ~/data/nst/preprocessed_test --validation_data_path ~/data/nst/preprocessed_test --save_dir test_results --use_gpu --epochs 40 --batch_size 16 --train_with_augmentations --stored_model_path ~/home/models/dummy.pth
```

#### Finetuning DanSpeech models
```commandline
python finetune.py --model_id test_id --train_data_path ~/data/nst/preprocessed_test --validation_data_path ~/data/nst/preprocessed_test --save_dir test_results --use_gpu --epochs 40 --batch_size 16 --train_with_augmentations --danspeech_model DanSpeechPrimary 
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
