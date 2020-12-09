import argparse

from args_parsing import add_standard_train_arguments, add_augmentation_arguments, add_training_parameters, \
    add_continue_training_parameters, add_multi_gpu_parameters
from deepspeech.train import _train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_standard_train_arguments(parser)
    add_augmentation_arguments(parser)
    add_training_parameters(parser)
    add_continue_training_parameters(parser)
    add_multi_gpu_parameters(parser)

    args = parser.parse_args()

    _train_model(model_id=args.model_id,
                 train_data_paths=args.train_data_paths,
                 train_data_weights=args.train_data_weights,
                 validation_data_path=args.validation_data_path,
                 cuda=args.use_gpu,
                 epochs=args.epochs,
                 num_freeze_layers=args.number_layers_freeze,
                 batch_size=args.batch_size,
                 save_dir=args.save_dir,
                 use_tensorboard=not args.no_tensorboard,
                 stored_model=args.continue_from_path,
                 augmented_training=args.train_with_augmentations or args.augmentation_list,
                 augmentations=args.augmentation_list,
                 lr=args.lr,
                 momentum=args.momentum,
                 max_norm=args.max_norm,
                 learning_anneal=args.learning_anneal,
                 rank=args.rank,
                 gpu_rank=args.gpu_rank,
                 world_size=args.world_size,
                 dist_backend=args.dist_backend,
                 dist_url=args.dist_url,
                 distributed=args.gpu_rank is not None,
                 save_every_epoch=args.save_every_epoch,
                 num_workers=args.number_of_data_workers,
                 continue_train=True
                 )
