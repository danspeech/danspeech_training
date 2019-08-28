from deepspeech.train import train_new
from deepspeech.train import continue_training
from deepspeech.train import finetune

from danspeech.deepspeech.model import DeepSpeech

if __name__ == '__main__':
    #model = DeepSpeech()

    # Begin training
    train_new(model_id=None, train_data_path='/scratch/s134843/danspeech/', validation_data_path='/scratch/s134843/danspeech/')
