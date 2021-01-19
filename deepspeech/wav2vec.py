

def load_wav2_vec():
    import fairseq
    w2v_path = "/home/rafje/trained_models/speech_recognition/fairseq/dan2vec/finetuned/no_normalization/checkpoint_11-01_fixed.pt"
    # w2v_path = "/home/rafje/trained_models/speech_recognition/fairseq/dan2vec/finetuned/no_normalization/best_fixed.pt"
    dict_path = "/home/rafje/data/nst/manifest"
    model, cfg = fairseq.checkpoint_utils.load_model_ensemble(
        [w2v_path],
        arg_overrides={"data": dict_path}
    )
    model = model[0]
    model.make_generation_fast_(
        beamable_mm_beam_size=4,
        need_attn=False,
    )
    return model