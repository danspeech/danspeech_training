import argparse

from deepspeech.train import _train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Standard where to save etc.
    parser.add_argument('--model_id', type=str, help='Id of model.')
    parser.add_argument('--train_data_path', type=str, help='Path to folder where training data is located.')
    parser.add_argument('--validation_data_path', type=str, help='Path to folder where training data is located.')
    parser.add_argument('--save_dir', type=str, help='Path to where model and tensorboard logs are saved.')
    parser.add_argument('--no_tensorboard', action='store_true', help='Whether to use tensorboard to track training')

    # Augmentations
    parser.add_argument('--train_with_augmentations', action='store_true', help='Whether to train with augmentations.',
                        default=False)
    parser.add_argument('--default_augmentations', action='store_true',
                        help='Train using default danspeech augmentations.', default=True)
    parser.add_argument('--augmentation_list', nargs='+',
                        help='Name of augmentations to use. If not given, then all danspeech augmentations are used',
                        default=None)

    parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU', default=False)

    # Training properties
    parser.add_argument('--epochs', type=int, help="Number of epochs to train", default=50)
    parser.add_argument('--batch_size', type=int, help="Number of epochs to train", default=64)
    parser.add_argument('--number_layers_freeze', type=int, help="How many layers to freeze during training.",
                        default=0)
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--learning_anneal', default=1.0, type=float,
                        help='Annealing applied to learning rate every epoch')

    # Use DanSpeech model
    parser.add_argument('--danspeech_model', type=str, help="Which DanSpeech model to finetune.", default=None)
    parser.add_argument('--stored_model_path', type=str,
                        help="Path to a stored model to finetune or continue training from", default=None)

    args = parser.parse_args()

    model = None
    if args.danspeech_model:
        import danspeech.pretrained_models as pretrained_models

        model = pretrained_models.get_model_from_string(args.danspeech_model)

    _train_model(model_id=args.model_id,
                 train_data_path=args.train_data_path,
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
                 finetune=True
                 )
