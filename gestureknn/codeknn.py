import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import paired_distances
import Levenshtein

from .args import get_configs
from .constant import UPPERBODY_PARENT, NUM_AUDIO_FEAT, NUM_BODY_FEAT, \
    NUM_MFCC_FEAT, NUM_JOINTS, STEP_SZ, WAV_TEST_SIZE, num_frames_code, num_frames, codebook_size, NUM_AUDIO_FEAT_FRAMES

args = get_configs()


def wavvq_distances(ls1, ls2, mode='sum'):
    if mode == 'sum':
        def ls2str(ls):
            ls = ls.reshape(NUM_AUDIO_FEAT_FRAMES, -1).transpose()  # (NUM_AUDIO_FEAT_FRAMES, groups=2)
            str1 = ''.join([chr(int(i)) for i in ls[0]])
            str2 = ''.join([chr(int(i)) for i in ls[1]])
            return str1, str2

        ls1_group1_str, ls1_group2_str = ls2str(ls1)
        ls2_group1_str, ls2_group2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_group1_str, ls2_group1_str) + Levenshtein.distance(ls1_group2_str, ls2_group2_str)

    elif mode == 'combine':
        def ls2str(ls):
            ls = ls.reshape(-1, 2).transpose()  # (NUM_AUDIO_FEAT_FRAMES * 2, groups=2)
            ls = ls[0] * 320 + ls[1]
            str = ''.join([chr(int(i)) for i in ls])
            return str

        ls1_str = ls2str(ls1)
        ls2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_str, ls2_str)


class CodeKNN(object):
    def __init__(self, mfcc_train, code_train, feat_train, wavlm_train, wavlm_train_feat, speech_features,
                 speech_features_feat, wavvq_train_feat, phase_train, context_train,
                 use_wavlm=False, use_wavvq=False, use_phase=False, use_txt=False):
        super(CodeKNN, self).__init__()

        # mfcc_train shape    : (num_seq, num_frames=240, NUM_MFCC_FEAT=13)
        # code_train shape    : (num_seq, num_frames_code=30)

        if use_wavlm:
            self.step_sz = wavlm_train.shape[1] // num_frames_code
            self.n_db_seq = wavlm_train.shape[0]
            self.n_db_frm = wavlm_train.shape[1]
        elif use_wavvq:
            self.step_sz = 398 / num_frames_code
            self.n_db_seq = wavvq_train_feat.shape[0]
            self.n_db_frm = 398
        else:
            self.step_sz = num_frames // num_frames_code  # 8
            self.n_db_seq = mfcc_train.shape[0]
            self.n_db_frm = mfcc_train.shape[1]
        print('step_sz is ', self.step_sz)
        self.mfcc_train = mfcc_train
        self.code_train = code_train
        self.feat_train = feat_train
        self.wavlm_train = wavlm_train
        self.wavlm_train_feat = wavlm_train_feat
        self.wavvq_train_feat = wavvq_train_feat
        self.phase_train = phase_train
        self.phase_channels = 8
        self.context_train = context_train
        # self.energy = energy
        # self.pitch = pitch
        # self.volume = volume
        self.speech_features = speech_features
        self.speech_features_feat = speech_features_feat
        self.code_to_signature()
        self.code_to_freq()
        self.use_phase = use_phase

    def init_code_phase(self):
        init_i = np.random.randint(0, self.n_db_seq)
        init_j = np.random.randint(0, self.n_db_frm - int(num_frames / num_frames_code))
        init_code = self.code_train[init_i, init_j // num_frames_code]
        if not self.use_phase:
            return init_code
        else:
            init_phase = self.phase_train[init_i, init_j:init_j + int(num_frames / num_frames_code)]
            phase = torch.tensor([j.detach().cpu().numpy() for j in init_phase[:, 0]]).squeeze().squeeze().numpy()  # 32, 8
            amps = torch.tensor([j.detach().cpu().numpy() for j in init_phase[:, 2]]).squeeze().squeeze().numpy()  # 32, 8
            phase_amp = np.concatenate((phase, amps), axis=1)
            return init_code, phase_amp

    def code_to_signature(self):
        x = np.load(args.codebook_signature)['signature']
        self.c2s = {}
        for i in range(codebook_size):
            self.c2s[i] = x[i]

    def code_to_freq(self):
        from collections import Counter
        train_code = np.load(args.train_codebook)['code']
        code = train_code.flatten()
        result = Counter(code)
        result_sorted = sorted(result.items(), key=lambda item: item[1], reverse=True)
        x = []
        y = []
        for d in result_sorted:
            x.append(d[0])
            y.append(d[1])
        y = 1 - np.array(y) / sum(y)
        self.c2f = {}
        for i in range(codebook_size):
            if i in x:
                self.c2f[i] = y[x.index(i)]
            else:
                self.c2f[i] = 1
        self.freq_dist_cands = list(self.c2f.values())

    def search_code_knn(self, clip_test, desired_k, use_feature=False, use_wavlm=False, use_freq=False, seed_code=None,
                        use_wavvq=False, use_phase=False, seed_phase=None, use_txt=False, clip_context=None, use_aud=False):
        # mfcc_test shape : (num_frames=3600, NUM_MFCC_FEAT=13)
        pose_cands = []
        result = []
        result_phase = []
        vote = []
        if use_phase:
            if seed_code != None:
                init_code = seed_code
                init_phase_amp = seed_phase
            else:
                init_code, init_phase_amp = self.init_code_phase()
            result_phase.append(init_phase_amp)
            print(init_code, init_phase_amp.shape)
        else:
            if seed_code != None:
                init_code = seed_code
            else:
                init_code = self.init_code_phase()
                print(init_code)
        result.append(init_code)

        for code in self.c2s.keys():
            pose_cands.append(code)

        i = 0
        while i < len(clip_test):
            # for i in range(0, len(clip_test), STEP_SZ * self.step_sz):
            print(str(i) + '\r', end='')
            pos_dist_cands = []
            for code in self.c2s.keys():
                if code == result[-1]:  # avoid still in the same code, optical
                    pos_dist_cands.append(1e10000)
                    continue
                pos_dist_cands.append(np.linalg.norm(self.c2s[result[-1]] - self.c2s[code]))
            # pose_cands shape: (codebook_size, )       pos_dist_cands shape: (codebook_size, )

            # len(aud_dist_cands) = codebook_size
            pos_score = np.array(pos_dist_cands).argsort().argsort()

            use_freq = True
            if use_freq:  # control signal
                freq_score = np.array(self.freq_dist_cands).argsort().argsort()
                pos_score = pos_score + freq_score * 0.05  # 0.1

            if use_txt:
                if use_wavlm:
                    clip_context_ = clip_context[int(i / self.wavlm_train.shape[1] * 30)]
                elif use_wavvq:
                    clip_context_ = clip_context[int(i / 398 * 30)]
                txt_dist_cands, txt_index_cands, aux_ = self.search_text_cands(clip_context_)
                txt_score = np.array(txt_dist_cands).argsort().argsort()
                combined_score_ = pos_score + txt_score
                combined_sorted_idx_ = np.argsort(combined_score_).tolist()  # len=num_seq

            if use_aud:
                if use_wavvq and use_feature:
                    clip_wavvq = clip_test[int(i)]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_wavvq, mode='wavvq_feat')
                elif use_wavlm and not use_feature:
                    clip_wavlm = clip_test[i:i + self.step_sz]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_wavlm, mode='wavlm')
                elif use_wavlm and use_feature:
                    clip_wavlm_feat = clip_test[i]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_wavlm_feat, mode='wavlm_feat')
                elif not use_wavlm and use_feature:
                    clip_feat = clip_test[i]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_feat, mode='feat')
                elif not use_wavlm and not use_feature:
                    clip_mfcc = clip_test[i:i + self.step_sz]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_mfcc, mode='audio')

                aud_score = np.array(aud_dist_cands).argsort().argsort()
                combined_score = pos_score + aud_score
                combined_sorted_idx = np.argsort(combined_score).tolist()  # len=num_seq

            if not use_phase and use_txt and use_aud:
                combined_score = pos_score + aud_score + txt_score
                combined_sorted_idx = np.argsort(combined_score).tolist()
                if np.random.rand() > 0.5:
                    for ii in aud_index_cands[combined_sorted_idx[desired_k]]:
                        result.append(ii)
                else:
                    for ii in txt_index_cands[combined_sorted_idx[desired_k]]:
                        result.append(ii)
            elif not use_phase and use_aud:
                for ii in aud_index_cands[combined_sorted_idx[desired_k]]:
                    result.append(ii)
            elif not use_phase and use_txt:
                for ii in aud_index_cands[combined_sorted_idx_[desired_k]]:
                    result.append(ii)
            elif use_phase and use_aud and not use_txt:
                tmp_distance = []
                tmp_phase_amp = []
                for index in combined_sorted_idx[:2]:
                    candidates_index = aux[index]
                    candidates_phase = self.phase_train[candidates_index[0]][int(candidates_index[1] / 398 * 240):int(candidates_index[1] / 398 * 240) + 32]  # (32, 4, (1, 8, 1))
                    phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze().squeeze().numpy()  # 32, 8
                    amp = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze().squeeze().numpy()
                    phase_amp = np.concatenate((phase[:8], amp[:8]), axis=1)  # 32, 16
                    tmp_distance.append(paired_distances([np.concatenate((result_phase[-1][-5:], phase_amp[:3]), axis=0).reshape(-1)], [np.concatenate((result_phase[-1][-3:], phase_amp[:5]), axis=0).reshape(-1)], metric='cosine')[0])
                    tmp_phase_amp.append(np.concatenate((phase[-8:], amp[-8:]), axis=1))
                final_index = tmp_distance.index(min(tmp_distance))
                # print(final_index)
                for ii in aud_index_cands[combined_sorted_idx[final_index]]:
                    result.append(ii)
                result_phase.append(tmp_phase_amp[final_index])

            elif use_phase and not use_aud and use_txt:
                tmp_distance = []
                tmp_phase_amp = []
                for index in combined_sorted_idx_[:2]:
                    candidates_index = aux_[index]
                    candidates_phase = self.phase_train[candidates_index[0]][int(candidates_index[1] / 398 * 240):int(candidates_index[1] / 398 * 240) + 32]  # (32, 4, (1, 8, 1))
                    phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze().squeeze().numpy()  # 32, 8
                    amp = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze().squeeze().numpy()
                    phase_amp = np.concatenate((phase[:8], amp[:8]), axis=1)  # 32, 16
                    tmp_distance.append(paired_distances([np.concatenate((result_phase[-1][-5:], phase_amp[:3]), axis=0).reshape(-1)], [np.concatenate((result_phase[-1][-3:], phase_amp[:5]), axis=0).reshape(-1)], metric='cosine')[0])
                    tmp_phase_amp.append(np.concatenate((phase[-8:], amp[-8:]), axis=1))
                final_index = tmp_distance.index(min(tmp_distance))
                # print(final_index)
                for ii in txt_index_cands[combined_sorted_idx_[final_index]]:
                    result.append(ii)
                result_phase.append(tmp_phase_amp[final_index])

            elif use_phase and use_aud and use_txt:
                tmp_distance = []
                tmp_phase_amp = []
                for index in combined_sorted_idx[:1]:
                    candidates_index = aux[index]
                    candidates_phase = self.phase_train[candidates_index[0]][int(candidates_index[1] / 398 * 240):int(candidates_index[1] / 398 * 240) + 32]  # (32, 4, (1, 8, 1))
                    phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze().squeeze().numpy()  # 32, 8
                    amp = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze().squeeze().numpy()
                    phase_amp = np.concatenate((phase[:8], amp[:8]), axis=1)  # 32, 16
                    tmp_distance.append(paired_distances([np.concatenate((result_phase[-1][-5:], phase_amp[:3]), axis=0).reshape(-1)], [np.concatenate((result_phase[-1][-3:], phase_amp[:5]), axis=0).reshape(-1)], metric='cosine')[0])
                    tmp_phase_amp.append(np.concatenate((phase[-8:], amp[-8:]), axis=1))
                for index in combined_sorted_idx_[:1]:
                    candidates_index = aux_[index]
                    candidates_phase = self.phase_train[candidates_index[0]][int(candidates_index[1] / 398 * 240):int(candidates_index[1] / 398 * 240) + 32]  # (32, 4, (1, 8, 1))
                    phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze().squeeze().numpy()  # 32, 8
                    amp = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze().squeeze().numpy()
                    phase_amp = np.concatenate((phase[:8], amp[:8]), axis=1)  # 32, 16
                    tmp_distance.append(paired_distances([np.concatenate((result_phase[-1][-5:], phase_amp[:3]), axis=0).reshape(-1)], [np.concatenate((result_phase[-1][-3:], phase_amp[:5]), axis=0).reshape(-1)], metric='cosine')[0])
                    tmp_phase_amp.append(np.concatenate((phase[-8:], amp[-8:]), axis=1))
                final_index = tmp_distance.index(min(tmp_distance))
                # print(final_index)
                if final_index in [0]:
                    for ii in aud_index_cands[combined_sorted_idx[final_index]]:
                        result.append(ii)
                elif final_index in [1]:
                    for ii in txt_index_cands[combined_sorted_idx_[final_index - 1]]:
                        result.append(ii)
                else:
                    raise ValueError("wrong final index")
                result_phase.append(tmp_phase_amp[final_index])
                vote.append(final_index)
                # print(final_index)
            i += STEP_SZ * self.step_sz

        if use_phase:
            return np.array(result)[1:1 + num_frames_code], np.array(result_phase)[1:], np.array(vote)
        else:
            return np.array(result)[1:1 + num_frames_code], np.array(result_phase)[1:]

    def search_audio_cands(self, clip_input, mode='audio'):
        # mfcc_test:(num_frames=3600, NUM_MFCC_FEAT=13)
        aud_dist_cands = [1e+3] * codebook_size
        aud_index_cands = [[] for _ in range(codebook_size)]
        aux = [[] for _ in range(codebook_size)]
        for j in range(self.n_db_seq):
            k = 0
            while k < self.n_db_frm - STEP_SZ * self.step_sz:
                # for k in range(0, self.n_db_frm-STEP_SZ*self.step_sz, self.step_sz):
                code = self.code_train[j, int(k / self.step_sz)]
                if mode == 'wavvq_feat':
                    audio_sim_score = wavvq_distances(clip_input, self.wavvq_train_feat[j, int(k)], mode='combine')
                elif mode == 'audio':
                    audio_sim_score = paired_distances([clip_input.reshape(-1)], [self.mfcc_train[j, k:k + self.step_sz].reshape(-1)], metric='cosine')[0]
                elif mode == 'feat':
                    audio_sim_score = paired_distances([clip_input], [self.feat_train[j, k]], metric='cosine')[0]
                elif mode == 'wavlm':
                    audio_sim_score = paired_distances([clip_input.reshape(-1)], [self.wavlm_train[j, k:k + self.step_sz].reshape(-1)], metric='cosine')[0]
                elif mode == 'wavlm_feat':
                    audio_sim_score = paired_distances([clip_input], [self.wavlm_train_feat[j, k]], metric='cosine')[0]
                if audio_sim_score < aud_dist_cands[code]:
                    aud_dist_cands[code] = audio_sim_score
                    aud_index_cands[code] = self.code_train[j, int(k / self.step_sz):int(k / self.step_sz) + STEP_SZ]
                    aux[code] = [j, int(k)]
                k += self.step_sz
        return aud_dist_cands, aud_index_cands, aux

    def search_code_change(self, clip_test):
        '''
        energy  (3600,)
        pitch   (3600,)
        volume  (3600,)
        '''
        result = []
        # init_code = self.init_code()
        init_code = 34
        result.append(init_code)
        test_energy = clip_test[0]
        test_pitch = clip_test[1]
        test_volume = clip_test[2]
        pdb.set_trace()

    def search_text_cands(self, clip_input, mode='wavvq_feat'):
        txt_dist_cands = [1e+3] * codebook_size
        txt_index_cands = [[] for _ in range(codebook_size)]
        aux = [[] for _ in range(codebook_size)]
        for j in range(self.n_db_seq):
            for k in range(0, 240 - STEP_SZ * 8, 8):
                code = self.code_train[j, k // 8]
                if mode == 'wavvq_feat':
                    text_sim_score = paired_distances([clip_input], [self.context_train[j, k // 8]], metric='cosine')[0]
                if text_sim_score < txt_dist_cands[code]:
                    txt_dist_cands[code] = text_sim_score
                    txt_index_cands[code] = self.code_train[j, (k // 8):(k // 8) + STEP_SZ]
                    aux[code] = [j, k]
        return txt_dist_cands, txt_index_cands, aux
