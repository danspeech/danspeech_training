import argparse

from args_parsing import add_standard_train_arguments, add_augmentation_arguments, add_training_parameters, \
    add_finetune_parameters, add_multi_gpu_parameters
from deepspeech.train import _train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_standard_train_arguments(parser)
    add_augmentation_arguments(parser)
    add_training_parameters(parser)
    add_finetune_parameters(parser)
    add_multi_gpu_parameters(parser)

    args = parser.parse_args()

    model = None
    if args.danspeech_model:
        import danspeech.pretrained_models as pretrained_models

        model = pretrained_models.get_model_from_string(args.danspeech_model)

    _train_model(model_id=args.model_id,
                 train_data_paths=args.train_data_paths,
                 train_data_weights=args.train_data_weights,
                 validation_data_path=args.validation_data_path,
                 cuda=args.use_gpu,
                 epochs=args.epochs,
                 num_freeze_layers=args.number_layers_freeze,
                 batch_size=args.batch_size,
                 danspeech_model=model,
                 stored_model=args.stored_model_path,
                 save_dir=args.save_dir,
                 lr=args.lr,
                 momentum=args.momentum,
                 max_norm=args.max_norm,
                 learning_anneal=args.learning_anneal,
                 use_tensorboard=not args.no_tensorboard,
                 augmented_training=args.train_with_augmentations or args.augmentation_list,
                 augmentations=args.augmentation_list,
                 rank=args.rank,
                 gpu_rank=args.gpu_rank,
                 world_size=args.world_size,
                 dist_backend=args.dist_backend,
                 dist_url=args.dist_url,
                 distributed=args.gpu_rank is not None,
                 save_every_epoch=args.save_every_epoch,
                 finetune=True
                 )
