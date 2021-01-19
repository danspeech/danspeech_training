import os

import torch
from danspeech.deepspeech.decoder import GreedyDecoder
from danspeech.deepspeech.model import DeepSpeech
from tqdm import tqdm

from audio.datasets import DanSpeechDataset, BatchDataLoader, Dan2VecInferenceDataset, Wav2VecBatchDataLoader
from audio.parsers import SpectrogramAudioParser
from deepspeech.wav2vec import load_wav2_vec


def test_model(model_path, data_path, decoder="greedy", cuda=False, batch_size=96, num_workers=4,
               lm_path=None, alpha=1.3, beta=0.4, cutoff_top_n=40, cutoff_prob=1.0, beam_width=64,
               lm_workers=4, verbose=False, transcriptions_out_file=None, wav2vec=False):
    torch.set_grad_enabled(False)

    labels = None
    if wav2vec:
        from fairseq.data import Dictionary
        model = load_wav2_vec()
        manifest_path = "/home/rafje/data/nst/manifest"
        dict_path = os.path.join(manifest_path, "dict.ltr.txt")
        target_dict = Dictionary.load(dict_path)
        target_dict.symbols[4] = " "
        labels = target_dict.symbols
    else:
        model = DeepSpeech.load_model(model_path)
        labels = model.labels

    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)
    model.eval()

    if decoder == "beam":
        from danspeech.deepspeech.decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=lm_path, alpha=alpha, beta=beta,
                                 cutoff_top_n=cutoff_top_n, cutoff_prob=cutoff_prob,
                                 beam_width=beam_width, num_processes=lm_workers)
    elif decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=0)
    else:
        raise AttributeError("please specify a valid decoder, DanSpeech currently supports [greedy, beam]")

    target_decoder = GreedyDecoder(labels, blank_index=0)
    test_dataset = None
    if wav2vec:
        test_dataset = Dan2VecInferenceDataset(data_path, labels=labels)
        test_batch_loader = Wav2VecBatchDataLoader(test_dataset, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=False)
    else:
        test_parser = SpectrogramAudioParser(audio_config=model.audio_conf, data_augmenter=None)
        test_dataset = DanSpeechDataset(data_path, labels=model.labels, audio_parser=test_parser)
        test_batch_loader = BatchDataLoader(test_dataset, batch_size=batch_size,
                                            num_workers=num_workers, shuffle=False)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0

    if transcriptions_out_file:
        out_f = open(transcriptions_out_file, "w", encoding="utf-8")
        out_f.write("reference,transcription,WER,CER\n")

    for i, (data) in tqdm(enumerate(test_batch_loader), total=len(test_batch_loader)):
        if wav2vec:
            net_input, targets, target_sizes = data
            net_input["source"] = net_input["source"].to(device)
            net_input["padding_mask"] = net_input["padding_mask"].to(device)
            out = model.get_normalized_probs(model(**net_input), False).transpose(0, 1)
            decoded_output, _ = decoder.decode(out)
        else:
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
            out, output_sizes = model(inputs, input_sizes)
            decoded_output, _ = decoder.decode(out.data, output_sizes.data)

        split_targets = []
        offset = 0
        targets = targets.numpy()
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        target_strings = target_decoder.convert_to_strings(split_targets)
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            if wav2vec:
                transcript = transcript[:-1]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference)
            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()), "CER:", float(cer_inst) / len(reference), "\n")

            if transcriptions_out_file:
                out_f.write("{0},{1},{2},{3}\n".format(reference.lower(), transcript.lower(),
                                                       float(wer_inst) / len(reference.split()),
                                                       float(cer_inst) / len(reference)))

    if transcriptions_out_file:
        out_f.close()

    if decoder is not None:
        wer = float(total_wer) / num_tokens
        cer = float(total_cer) / num_chars

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
