import argparse

from deepspeech.test import test_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, help='Path to model to evaluate.')
    parser.add_argument('--test_dataset', type=str, help='Path to test dataset')
    parser.add_argument('--transcriptions_out_file', type=str, help="File for to output transcriptions in.")

    parser.add_argument('--use_gpu', action="store_true", help="Whether to use GPU.")
    parser.add_argument('--batch_size', default=96, type=int, help='Batch size ')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")

    parser.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
    parser.add_argument('--decoder', default='greedy', type=str, help='beam og greedy')
    parser.add_argument('--lm_path', default=None, type=str,
                        help='Path to an (optional) kenlm language model for use with beam search')
    parser.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
    parser.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
    parser.add_argument('--cutoff_top_n', default=40, type=int,
                        help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                             'vocabulary will be used in beam search, default 40.')
    parser.add_argument('--cutoff_prob', default=1.0, type=float,
                        help='Cutoff probability in pruning,default 1.0, no pruning.')
    parser.add_argument('--lm_workers', default=1, type=int, help='Number of LM processes to use')
    parser.add_argument('--use_wav2vec', action="store_true", help = "Whether to use wav2vec model")

    args = parser.parse_args()

    test_model(model_path=args.model_path, data_path=args.test_dataset, decoder=args.decoder, cuda=args.use_gpu,
               batch_size=args.batch_size, num_workers=args.num_workers, lm_path=args.lm_path, alpha=args.alpha,
               beta=args.beta, cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob, beam_width=args.beam_width,
               lm_workers=args.lm_workers, verbose=args.verbose, transcriptions_out_file=args.transcriptions_out_file,
               wav2vec=args.use_wav2vec)
