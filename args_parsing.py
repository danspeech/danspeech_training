def add_standard_train_arguments(parser):
    """
    Parameters such as model_id, data and where to save mode.
    """
    parser.add_argument('--model_id', type=str, help='Id of model.')
    parser.add_argument('--train_data_path', type=str, help='Path to folder where training data is located.')
    parser.add_argument('--validation_data_path', type=str, help='Path to folder where training data is located.')
    parser.add_argument('--save_dir', type=str, help='Path to where model and tensorboard logs are saved.')
    parser.add_argument('--no_tensorboard', action='store_true', help='Whether to use tensorboard to track training')
    parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU', default=False)


def add_augmentation_arguments(parser):
    # Augmentations
    parser.add_argument('--train_with_augmentations', action='store_true', help='Whether to train with augmentations.',
                        default=False)
    parser.add_argument('--augmentation_list', nargs='+',
                        help='Name of augmentations to use. If not given, then all danspeech augmentations are used',
                        default=None)


def add_training_parameters(parser):
    # Training parameters
    parser.add_argument('--epochs', type=int, help="Number of epochs to train", default=50)
    parser.add_argument('--batch_size', type=int, help="Number of epochs to train", default=64)
    parser.add_argument('--number_layers_freeze', type=int, help="How many layers to freeze during training.",
                        default=0)
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--learning_anneal', default=1.0, type=float,
                        help='Annealing applied to learning rate every epoch')


def add_audio_parameters(parser):
    # Audio parameters
    parser.add_argument('--sampling_rate', default=16000, type=int, help='Sample rate')
    parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
    parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
    parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')


def add_neural_network_parameters(parser):
    # Neural network parameters
    parser.add_argument('--hidden_size', default=800, type=int, help='Hidden size of RNNs')
    parser.add_argument('--hidden_layers', default=5, type=int, help='Number of RNN layers')
    parser.add_argument('--rnn_type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--no_bidirectional', action='store_false', default=True,
                        help='Turn off bi-directional RNNs, introduces lookahead convolution')


def add_continue_training_parameters(parser):
    parser.add_argument('--continue_from_path', type=str, help='Path to model to continue training from.')


def add_finetune_parameters(parser):
    # Use DanSpeech model
    parser.add_argument('--danspeech_model', type=str, help="Which DanSpeech model to finetune.", default=None)
    parser.add_argument('--stored_model_path', type=str,
                        help="Path to a stored model to finetune or continue training from", default=None)
