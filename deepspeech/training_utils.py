import os
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def to_np(x):
    return x.data.cpu().numpy()


class TensorBoardLogger(object):

    def __init__(self, id, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)

    def update(self, epoch, values):
        values = {
            'Avg. Train Loss': values["loss_results"][epoch],
            'WER': values["wer"][epoch],
            'CER': values["cer"][epoch]
        }

        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)

    def load_previous_values(self, start_epoch, values):
        for i in range(start_epoch):
            values = {
                'Avg. Train Loss': values["loss_results"][i],
                'WER': values["wer"][i],
                'CER': values["cer"][i]
            }

            self.tensorboard_writer.add_scalars(self.id, values, i + 1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def get_audio_config(normalize=True, sample_rate=16000, window="hamming", window_stride=0.01, window_size=0.02):
    return {
        "normalize": normalize,
        "sample_rate": sample_rate,
        "window": window,
        "window_stride": window_stride,
        "window_size": window_size
    }


def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
              cer_results=None, wer_results=None, avg_loss=None, meta=None, distributed=False,
              streaming_model=None, context=None):

    supported_rnns = {
        'lstm': nn.LSTM,
        'rnn': nn.RNN,
        'gru': nn.GRU
    }
    supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

    if distributed:
        package = {
            'model_name': model.module.model_name,
            'conv_layers': model.module.conv_layers,
            'rnn_hidden_size': model.module.rnn_hidden_size,
            'rnn_layers': model.module.rnn_layers,
            'rnn_type': supported_rnns_inv.get(model.module.rnn_type, model.module.rnn_type.__name__.lower()),
            'audio_conf': model.module.audio_conf,
            'labels': model.module.labels,
            'bidirectional': model.module.bidirectional,
        }
    else:
        package = {
            'model_name': model.model_name,
            'conv_layers': model.conv_layers,
            'rnn_hidden_size': model.rnn_hidden_size,
            'rnn_layers': model.rnn_layers,
            'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()),
            'audio_conf': model.audio_conf,
            'labels': model.labels,
            'state_dict': model.state_dict(),
            'bidirectional': model.bidirectional
        }

    if optimizer is not None:
        package['optim_dict'] = optimizer.state_dict()
    if avg_loss is not None:
        package['avg_loss'] = avg_loss
    if epoch is not None:
        package['epoch'] = epoch + 1
    if iteration is not None:
        package['iteration'] = iteration
    if loss_results is not None:
        package['loss_results'] = loss_results
        package['cer_results'] = cer_results
        package['wer_results'] = wer_results
    if meta is not None:
        package['meta'] = meta
    if streaming_model is not None:
        package['streaming_model'] = streaming_model
    if context is not None:
        package['context'] = context
    return package
