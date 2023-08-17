import numpy as np
import torch
from tqdm import tqdm
import os
import Levenshtein
import pdb

from .args import get_configs
from .utils import normalize_data
from .data_processing import load_db_codebook, calc_data_stats
from .codeknn import CodeKNN

args = get_configs()


def predict_code_from_audio(train_mfcc, train_code, test_mfcc, data_stats, train_feat, test_feat, train_wavlm, test_wavlm,
                            train_wavlm_feat, test_wavlm_feat, speech_features, test_speech_features,
                            train_speech_features_feat, test_speech_features_feat, train_wavvq_feat, test_wavvq_feat,
                            train_phase, test_phase, train_context, test_context,
                            use_feature=False, use_wavlm=False, use_freq=False, use_speechfeat=False, use_wavvq=False,
                            use_phase=False, use_txt=False, use_aud=False, frames=0):
    norm_mfcc_train = normalize_data(train_mfcc, data_stats['mfcc_train_mean'], data_stats['mfcc_train_std'])
    norm_mfcc_train = norm_mfcc_train.transpose((0, 2, 1))
    norm_mfcc_test = normalize_data(test_mfcc, data_stats['mfcc_train_mean'], data_stats['mfcc_train_std'])
    norm_mfcc_test = norm_mfcc_test.transpose((0, 2, 1))

    norm_feat_train = normalize_data(train_feat, data_stats['feat_train_mean'], data_stats['feat_train_std'])
    norm_feat_train = norm_feat_train.transpose((0, 2, 1))
    norm_feat_test = normalize_data(test_feat, data_stats['feat_train_mean'], data_stats['feat_train_std'])
    norm_feat_test = norm_feat_test.transpose((0, 2, 1))

    n_test_seq = frames if frames != 0 else test_wavvq_feat.shape[0]  # test_mfcc.shape[0]

    train_wavlm = train_wavlm.transpose((0, 2, 1))
    test_wavlm = test_wavlm.transpose((0, 2, 1))
    train_wavlm_feat = train_wavlm_feat.transpose((0, 2, 1))
    test_wavlm_feat = test_wavlm_feat.transpose((0, 2, 1))

    # norm_energy, _, _ = normalize_feat(energy)
    # norm_pitch, _, _ = normalize_feat(pitch)
    # norm_volume, _, _ = normalize_feat(volume)
    #
    # norm_energy_test, _, _ = normalize_feat(test_energy)
    # norm_pitch_test, _, _ = normalize_feat(test_pitch)
    # norm_volume_test, _, _ = normalize_feat(test_volume)

    norm_speech_features = normalize_data(speech_features, data_stats['speech_features_train_mean'], data_stats['speech_features_train_std'])
    norm_speech_features = norm_speech_features.transpose((0, 2, 1))
    norm_test_speech_features = normalize_data(test_speech_features, data_stats['speech_features_train_mean'], data_stats['speech_features_train_std'])
    norm_test_speech_features = norm_test_speech_features.transpose((0, 2, 1))

    norm_speech_features_feat = normalize_data(train_speech_features_feat, data_stats['speech_features_feat_train_mean'], data_stats['speech_features_feat_train_std'])
    norm_speech_features_feat = norm_speech_features_feat.transpose((0, 2, 1))
    norm_test_speech_features_feat = normalize_data(test_speech_features_feat, data_stats['speech_features_feat_train_mean'], data_stats['speech_features_feat_train_std'])
    norm_test_speech_features = norm_test_speech_features_feat.transpose((0, 2, 1))

    train_wavvq_feat = train_wavvq_feat.transpose((0, 2, 1))
    test_wavvq_feat = test_wavvq_feat.transpose((0, 2, 1))

    train_phase = train_phase.transpose((0, 2, 1))
    # test_phase = test_phase.transpose((0, 2, 1))

    train_context = train_context.transpose((0, 2, 1))
    test_context = test_context.transpose((0, 2, 1))

    gesture_knn = CodeKNN(mfcc_train=norm_mfcc_train, code_train=train_code, feat_train=norm_feat_train,
                          wavlm_train=train_wavlm, wavlm_train_feat=train_wavlm_feat, speech_features=norm_speech_features,
                          speech_features_feat=norm_speech_features_feat, wavvq_train_feat=train_wavvq_feat,
                          phase_train=train_phase, context_train=train_context,
                          use_wavlm=use_wavlm, use_wavvq=use_wavvq, use_phase=use_phase, use_txt=use_txt)

    motion_output = []
    phase_output = []
    vote_output = []

    print('begin search...')
    for i in tqdm(range(0, n_test_seq)):
        # if use_speechfeat:
        #     pred_motion = gesture_knn.search_code_change(clip_test=[norm_energy_test[i], norm_pitch_test[i], norm_volume_test[i]])
        clip_context = test_context[i] if use_txt else None
        if use_wavvq and use_feature:
            if use_phase and use_aud:
                pred_motion, pred_phase, vote = gesture_knn.search_code_knn(clip_test=test_wavvq_feat[i], desired_k=args.desired_k, use_wavlm=False, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None, use_wavvq=True, use_phase=use_phase, seed_phase=phase_output[-1][-1] if i > 0 else None, use_txt=use_txt, clip_context=clip_context, use_aud=use_aud)
            elif not use_phase and use_aud:
                pred_motion, pred_phase = gesture_knn.search_code_knn(clip_test=test_wavvq_feat[i], desired_k=args.desired_k, use_wavlm=False, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None, use_wavvq=True, use_phase=use_phase, use_txt=use_txt, clip_context=clip_context, use_aud=use_aud)
            elif use_phase and not use_aud:
                pred_motion, pred_phase, vote = gesture_knn.search_code_knn(clip_test=test_wavvq_feat[i], desired_k=args.desired_k, use_wavlm=False, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None, use_wavvq=True, use_phase=use_phase, seed_phase=phase_output[-1][-1] if i > 0 else None, use_txt=use_txt, clip_context=clip_context, use_aud=use_aud)
        elif use_wavlm and not use_feature:
            pred_motion = gesture_knn.search_code_knn(clip_test=test_wavlm[i], desired_k=args.desired_k, use_wavlm=True, use_feature=False, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None)
        elif use_wavlm and use_feature:
            if use_phase and use_aud and use_txt:
                pred_motion, pred_phase, vote = gesture_knn.search_code_knn(clip_test=test_wavlm_feat[i], desired_k=args.desired_k, use_wavlm=True, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None, use_wavvq=False, use_phase=use_phase, seed_phase=phase_output[-1][-1] if i > 0 else None, use_txt=use_txt, clip_context=clip_context, use_aud=use_aud)
            else:
                pred_motion = gesture_knn.search_code_knn(clip_test=test_wavlm_feat[i], desired_k=args.desired_k, use_wavlm=True, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None)
        elif not use_wavlm and use_feature:
            pred_motion = gesture_knn.search_code_knn(clip_test=norm_feat_test[i], desired_k=args.desired_k, use_wavlm=False, use_feature=True, use_freq=use_freq)
        elif not use_wavlm and not use_feature:
            pred_motion = gesture_knn.search_code_knn(clip_test=norm_mfcc_test[i], desired_k=args.desired_k, use_wavlm=False, use_feature=False, use_freq=use_freq)
        print(pred_motion)
        motion_output.append(pred_motion)
        phase_output.append(pred_phase)
        # vote_output.append(vote)
        # print(np.array(pred_phase).shape)
    # np.savez_compressed('vote.npz', vote=np.array(vote_output))
    return np.array(motion_output)


def gestureknn():
    # (num_seq, NUM_MFCC_FEAT, num_frames=240), (num_seq, num_frames_code=30), (num_seq, NUM_MFCC_FEAT, num_frames=3600)
    train_mfcc, train_code, test_mfcc, train_feat, test_feat, train_wavlm, test_wavlm, train_wavlm_feat, \
        test_wavlm_feat, speech_features, test_speech_features, train_speech_features_feat, test_speech_features_feat, \
        train_wavvq_feat, test_wavvq_feat, train_phase, test_phase, train_context, test_context \
        = load_db_codebook(args.train_database, args.train_codebook, args.test_data, args.train_wavlm, args.test_wavlm, args.train_wavvq, args.test_wavvq)
    mfcc_train_mean, mfcc_train_std, _, _ = calc_data_stats(train_mfcc.transpose((0, 2, 1)), test_mfcc.transpose((0, 2, 1)))
    feat_train_mean, feat_train_std, _, _ = calc_data_stats(train_feat.transpose((0, 2, 1)), test_feat.transpose((0, 2, 1)))
    speech_features_train_mean, speech_features_train_std, _, _ = calc_data_stats(speech_features.transpose((0, 2, 1)), test_speech_features.transpose((0, 2, 1)))
    speech_features_feat_train_mean, speech_features_feat_train_std, _, _ = calc_data_stats(train_speech_features_feat.transpose((0, 2, 1)), test_speech_features_feat.transpose((0, 2, 1)))
    data_stats = {
        'mfcc_train_mean': mfcc_train_mean,
        'mfcc_train_std': mfcc_train_std,
        'feat_train_mean': feat_train_mean,
        'feat_train_std': feat_train_std,
        'speech_features_train_mean': speech_features_train_mean,
        'speech_features_train_std': speech_features_train_std,
        'speech_features_feat_train_mean': speech_features_feat_train_mean,
        'speech_features_feat_train_std': speech_features_feat_train_std
    }
    pred_seqs = predict_code_from_audio(train_mfcc, train_code, test_mfcc, data_stats, train_feat, test_feat, train_wavlm, test_wavlm,
                                        train_wavlm_feat, test_wavlm_feat, speech_features, test_speech_features,
                                        train_speech_features_feat, test_speech_features_feat, train_wavvq_feat, test_wavvq_feat,
                                        train_phase, test_phase, train_context, test_context,
                                        use_feature=True, use_wavlm=True, use_freq=False, use_speechfeat=False,
                                        use_wavvq=False, use_phase=True, use_txt=True, use_aud=True, frames=args.max_frames)  # if use wavlm, frames should be 15, and test_data should be 240
    print("pred_seqs.shape", pred_seqs.shape)
    np.savez_compressed(args.out_knn_filename, knn_pred=pred_seqs)

    print("Saved done!")
