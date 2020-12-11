import os
import time
from tqdm import tqdm
import warnings
import json

import torch
# warpctc bindings for pytorch can be found here: https://github.com/SeanNaren/warp-ctc
from warpctc_pytorch import CTCLoss

from audio.datasets import BatchDataLoader, DanSpeechDataset, DanSpeechMultiDataset, MultiDatasetBatchDataLoader, \
    DistributedWeightedSamplerCustom, DistributedSamplerCustom
from audio.parsers import SpectrogramAudioParser
from audio.augmentation import DanSpeechAugmenter
from danspeech.deepspeech.model import DeepSpeech, supported_rnns
from danspeech.deepspeech.decoder import GreedyDecoder
from deepspeech.training_utils import TensorBoardLogger, AverageMeter, reduce_tensor, sum_tensor, \
    get_audio_config, serialize
from danspeech.errors.training_errors import ArgumentMissingForOption


class NoModelSaveDirSpecified(Warning):
    pass


class NoLoggingDirSpecified(Warning):
    pass


class CudaNotAvailable(Warning):
    pass


class NoModelNameSpecified(Warning):
    pass


class InfiniteLossReturned(Warning):
    pass


def _train_model(model_id=None, train_data_paths=None, train_data_weights=None, validation_data_path=None, epochs=20,
                 stored_model=None, save_dir=None, use_tensorboard=True, augmented_training=False, batch_size=32,
                 num_workers=6, cuda=False, lr=3e-4, momentum=0.9, weight_decay=1e-5, max_norm=400,
                 learning_anneal=1.0, context=20, finetune=False, continue_train=False, train_new=False,
                 num_freeze_layers=None, rnn_type='gru', conv_layers=2, rnn_hidden_layers=5, rnn_hidden_size=800,
                 bidirectional=True, distributed=False, gpu_rank=None, dist_backend='nccl', rank=0,
                 dist_url='tcp://127.0.0.1:1550', world_size=1, danspeech_model=None, augmentations=None,
                 sampling_rate=16000, window="hamming", window_stride=0.01, window_size=0.02, save_every_epoch=0):
    # set training device
    if augmentations is None:
        augmentations = []
    main_proc = rank == 0
    print("Is main proc: {}".format(main_proc))
    if cuda and not torch.cuda.is_available():
        warnings.warn("Specified GPU training but cuda is not available...", CudaNotAvailable)

    device = torch.device("cuda" if cuda else "cpu")

    # prepare directories for storage and logging.
    if not save_dir:
        warnings.warn("You did not specify a directory for saving the trained model.\n"
                      "Defaulting to ~/.danspeech/custom/ directory.", NoModelSaveDirSpecified)

        save_dir = os.path.join(os.path.expanduser('~'), '.danspeech/models/')

    os.makedirs(save_dir, exist_ok=True)

    if not model_id:
        warnings.warn("You did not specify a name for the trained model.\n"
                      "Defaulting to danish_speaking_panda.pth", NoModelNameSpecified)

        model_id = "danish_speaking_panda"

    assert train_data_paths, "please specify path(s) to a valid directory with training data"
    assert validation_data_path, "please specify path to a valid directory with validation data"

    if main_proc and use_tensorboard:
        logging_process = True
        tensorboard_logger = TensorBoardLogger(model_id, save_dir)
    else:
        logging_process = False
        if main_proc:
            warnings.warn(
                "You did not specify a directory for logging training process. Training process will not be logged.",
                NoLoggingDirSpecified)

    # handle distributed processing
    if distributed:
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler
        from torch.nn.parallel import DistributedDataParallel

        if gpu_rank:
            torch.cuda.set_device(int(gpu_rank))

        dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                world_size=world_size, rank=rank)

    # initialize training metrics
    loss_results = torch.Tensor(epochs)
    cer_results = torch.Tensor(epochs)
    wer_results = torch.Tensor(epochs)

    # initialize helper variables
    avg_loss = 0
    start_epoch = 0
    start_iter = 0

    # load and initialize model metrics based on wrapper function
    if train_new:
        with open(os.path.dirname(os.path.realpath(__file__)) + '/labels.json', "r", encoding="utf-8") as label_file:
            labels = str(''.join(json.load(label_file)))

        # changing the default audio config is highly experimental, make changes with care and expect vastly
        # different results compared to baseline
        audio_conf = get_audio_config(normalize=True, sample_rate=sampling_rate, window=window,
                                      window_stride=window_stride, window_size=window_size)

        rnn_type = rnn_type.lower()
        conv_layers = conv_layers
        assert rnn_type in ["lstm", "rnn", "gru"], "rnn_type should be either lstm, rnn or gru"
        assert conv_layers in [1, 2, 3], "conv_layers must be set to either 1, 2 or 3"
        model = DeepSpeech(model_name=model_id,
                           conv_layers=conv_layers,
                           rnn_hidden_size=rnn_hidden_size,
                           rnn_layers=rnn_hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns.get(rnn_type),
                           audio_conf=audio_conf,
                           bidirectional=bidirectional,
                           streaming_inference_model=False,
                           # streaming inference should always be disabled during training
                           context=context)
        model = model.to(device)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=lr,
                                    momentum=momentum, nesterov=True, weight_decay=1e-5)

    if finetune:
        if not stored_model and danspeech_model is None:
            raise ArgumentMissingForOption("If you want to finetune, please provide the absolute path"
                                           "to a trained pytorch model object as the --continue_model_path argument")
        else:
            if danspeech_model:
                print("Using DanSpeech model: {}".format(danspeech_model.model_name))
                model = danspeech_model
            else:
                print("Loading checkpoint model %s" % stored_model)
                package = torch.load(stored_model, map_location=lambda storage, loc: storage)
                model = DeepSpeech.load_model_package(package)

            if num_freeze_layers:
                # freezing layers might result in unexpected results, use with cation
                model.freeze_layers(num_freeze_layers)

            model = model.to(device)

            parameters = model.parameters()
            optimizer = torch.optim.SGD(parameters, lr=lr,
                                        momentum=momentum, nesterov=True, weight_decay=1e-5)

    if continue_train:
        # continue_training wrapper
        if not stored_model:
            raise ArgumentMissingForOption("If you want to continue training, please support a package with previous"
                                           "training information or use the finetune option instead")
        else:
            print("Loading checkpoint model %s" % stored_model)
            package = torch.load(stored_model, map_location=lambda storage, loc: storage)
            model = DeepSpeech.load_model_package(package)
            model = model.to(device)
            # load stored training information
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                        nesterov=True, weight_decay=weight_decay)
            optim_state = package['optim_dict']
            optimizer.load_state_dict(optim_state)
            start_epoch = int(package['epoch']) + 1  # Index start at 0 for training

            print("Last successfully trained Epoch: {0}".format(start_epoch))

            start_epoch += 1
            start_iter = 0

            avg_loss = int(package.get('avg_loss', 0))
            loss_results_ = package['loss_results']
            cer_results_ = package['cer_results']
            wer_results_ = package['wer_results']

            # ToDo: Make depend on the epoch from the package
            previous_epochs = loss_results_.size()[0]
            print("Previously set to run for: {0} epochs".format(previous_epochs))

            loss_results[0:previous_epochs] = loss_results_
            wer_results[0:previous_epochs] = cer_results_
            cer_results[0:previous_epochs] = wer_results_

            if logging_process:
                tensorboard_logger.load_previous_values(start_epoch, package)

    # initialize DanSpeech augmenter
    if augmented_training:
        augmenter = DanSpeechAugmenter(sampling_rate=model.audio_conf["sample_rate"], augmentation_list=augmentations)
    else:
        augmenter = None

    # initialize audio parser and dataset
    # audio parsers
    training_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=augmenter)
    validation_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)

    # instantiate data-sets
    multi_data_set = False
    if len(train_data_paths) > 1:
        assert len(train_data_paths) == len(train_data_weights), "Must provide weights for each dataset"
        multi_data_set = True
        training_set = DanSpeechMultiDataset(train_data_paths, train_data_weights, labels=model.labels,
                                             audio_parser=training_parser)
    else:
        training_set = DanSpeechDataset(train_data_paths[0], labels=model.labels, audio_parser=training_parser)

    validation_set = DanSpeechDataset(validation_data_path, labels=model.labels, audio_parser=validation_parser)

    # initialize batch loaders
    if not distributed:
        # initialize batch loaders for single GPU or CPU training
        if multi_data_set:
            train_batch_loader = MultiDatasetBatchDataLoader(training_set, batch_size=batch_size,
                                                            num_workers=num_workers, pin_memory=True)
        else:
            train_batch_loader = BatchDataLoader(training_set, batch_size=batch_size, num_workers=num_workers,
                                                 shuffle=True, pin_memory=True)
        validation_batch_loader = BatchDataLoader(validation_set, batch_size=batch_size, num_workers=num_workers,
                                                  shuffle=False)
    else:
        # initialize batch loaders for distributed training on multiple GPUs
        if multi_data_set:
            train_sampler = DistributedWeightedSamplerCustom(training_set, num_replicas=world_size, rank=rank)
        else:
            train_sampler = DistributedSamplerCustom(training_set, num_replicas=world_size, rank=rank)

        train_batch_loader = BatchDataLoader(training_set, batch_size=batch_size,
                                             num_workers=num_workers,
                                             sampler=train_sampler,
                                             pin_memory=True)

        validation_sampler = DistributedSampler(validation_set, num_replicas=world_size, rank=rank)

        validation_batch_loader = BatchDataLoader(validation_set, batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  sampler=validation_sampler)

    decoder = GreedyDecoder(model.labels)
    criterion = CTCLoss()
    best_wer = None
    model = model.to(device)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[int(gpu_rank)])

    # verbatim training outputs during progress
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    print(model)
    print("Initializations complete, starting training pass on model: %s \n" % model_id)
    print("Number of parameters: %d \n" % DeepSpeech.get_param_size(model))
    try:
        for epoch in range(start_epoch, epochs):
            if distributed and epoch != 0:
                # distributed sampling, keep epochs on all GPUs
                train_sampler.set_epoch(epoch)

            print('started training epoch %d' % (epoch + 1))
            model.train()

            # timings per epoch
            end = time.time()
            start_epoch_time = time.time()
            num_updates = len(train_batch_loader)

            # per epoch training loop, iterate over all mini-batches in the training set
            for i, (data) in enumerate(train_batch_loader, start=start_iter):
                if i == num_updates:
                    break

                # grab and prepare a sample for a training pass
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                # measure data load times, this gives an indication on the number of workers required for latency
                # free training.
                data_time.update(time.time() - end)

                # parse data and perform a training pass
                inputs = inputs.to(device)

                # compute the CTC-loss and average over mini-batch
                out, output_sizes = model(inputs, input_sizes)
                out = out.transpose(0, 1)
                float_out = out.float()
                loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
                loss = loss / inputs.size(0)

                # check for diverging losses
                if distributed:
                    loss_value = reduce_tensor(loss, world_size).item()
                else:
                    loss_value = loss.item()

                if loss_value == float("inf") or loss_value == -float("inf"):
                    warnings.warn("received an inf loss, setting loss value to 0", InfiniteLossReturned)
                    loss_value = 0

                # update average loss, and loss tensor
                avg_loss += loss_value
                losses.update(loss_value, inputs.size(0))

                # compute gradients and back-propagate errors
                optimizer.zero_grad()
                loss.backward()

                # avoid exploding gradients by clip_grad_norm, defaults to 400
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                # stochastic gradient descent step
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (epochs), (i + 1), len(train_batch_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

                del loss, out, float_out

            # report epoch summaries and prepare validation run
            avg_loss /= len(train_batch_loader)
            loss_results[epoch] = avg_loss
            epoch_time = time.time() - start_epoch_time
            print('Training Summary Epoch: [{0}]\t'
                  'Time taken (s): {epoch_time:.0f}\t'
                  'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

            # prepare validation specific parameters, and set model ready for evaluation
            total_cer, total_wer = 0, 0
            model.eval()
            with torch.no_grad():
                for i, (data) in tqdm(enumerate(validation_batch_loader), total=len(validation_batch_loader)):
                    inputs, targets, input_percentages, target_sizes = data
                    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                    # unflatten targets
                    split_targets = []
                    offset = 0
                    targets = targets.numpy()
                    for size in target_sizes:
                        split_targets.append(targets[offset:offset + size])
                        offset += size

                    inputs = inputs.to(device)
                    out, output_sizes = model(inputs, input_sizes)
                    decoded_output, _ = decoder.decode(out, output_sizes)
                    target_strings = decoder.convert_to_strings(split_targets)

                    # compute accuracy metrics
                    wer, cer = 0, 0
                    for x in range(len(target_strings)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                        cer += decoder.cer(transcript, reference) / float(len(reference))

                    total_wer += wer
                    total_cer += cer
                    del out

            if distributed:
                # sums tensor across all devices if distributed training is enabled
                total_wer_tensor = torch.tensor(total_wer).to(device)
                total_wer_tensor = sum_tensor(total_wer_tensor)
                total_wer = total_wer_tensor.item()

                total_cer_tensor = torch.tensor(total_cer).to(device)
                total_cer_tensor = sum_tensor(total_cer_tensor)
                total_cer = total_cer_tensor.item()

                del total_wer_tensor, total_cer_tensor

            # compute average metrics for the validation pass
            avg_wer_epoch = (total_wer / len(validation_batch_loader.dataset)) * 100
            avg_cer_epoch = (total_cer / len(validation_batch_loader.dataset)) * 100

            # append metrics for logging
            loss_results[epoch] = avg_loss
            wer_results[epoch] = avg_wer_epoch
            cer_results[epoch] = avg_cer_epoch

            # log metrics for tensorboard
            if logging_process:
                logging_values = {
                    "loss_results": loss_results,
                    "wer": wer_results,
                    "cer": cer_results
                }
                tensorboard_logger.update(epoch, logging_values)

            # print validation metrics summary
            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, wer=avg_wer_epoch, cer=avg_cer_epoch))

            # check if the model is uni or bidirectional, and set streaming model accordingly
            if not bidirectional:
                streaming_inference_model = True
            else:
                streaming_inference_model = False

            # save model if it has the highest recorded performance on validation.
            if main_proc and (best_wer is None or best_wer > wer):
                model_path = save_dir + model_id + '.pth'
                print("Found better validated model, saving to %s" % model_path)
                torch.save(serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                     wer_results=wer_results, cer_results=cer_results,
                                     distributed=distributed, streaming_model=streaming_inference_model,
                                     context=context)
                           , model_path)
            if main_proc and save_every_epoch != 0 and (epoch + 1) % save_every_epoch == 0:
                model_epochs_save_path = save_dir + model_id + '_epoch_{}'.format(epoch + 1) + '.pth'
                print("Saving since save_every_epoch option has been given to %s" % model_epochs_save_path)
                torch.save(serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                     wer_results=wer_results, cer_results=cer_results,
                                     distributed=distributed, streaming_model=streaming_inference_model,
                                     context=context)
                           , model_epochs_save_path)

                param_groups = optimizer.param_groups
                if learning_anneal != 1.0:
                    for g in param_groups:
                        g['lr'] = g['lr'] / learning_anneal
                    print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

                best_wer = wer
                avg_loss = 0

            # reset start iteration for next epoch
            start_iter = 0

    except KeyboardInterrupt:
        print('Successfully exited training and stopped all processes.')
