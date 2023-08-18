import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict

from configs.parse_args import parse_args
from models.vqvae import VQVAE
from process.beat_data_to_lmdb import process_bvh
from process.bvh_to_position import bvh_to_npy
from process.process_bvh import make_bvh_GENEA2020_BT
from process.visualize_bvh import visualize


def visualize_code(args, model_path, save_path, prefix, code_source, normalize=True):
    # source_pose = source_pose[:3600]        # 60s, 60FPS

    # normalize
    if normalize:
        data_mean = np.array(args.data_mean).squeeze()
        data_std = np.array(args.data_std).squeeze()
        std = np.clip(data_std, a_min=0.01, a_max=None)

    with torch.no_grad():
        model = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        zs = [torch.from_numpy(code_source.flatten()).unsqueeze(0).to(mydevice)]
        pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()

        code.append(zs[0].squeeze(0).data.cpu().numpy())
        result.append(pose_sample)

    out_code = np.vstack(code)
    out_poses = np.vstack(result)

    if normalize:
        out_poses = np.multiply(out_poses, std) + data_mean
    print(out_poses.shape)
    print(out_code.shape)
    np.save(os.path.join(save_path, 'code' + prefix + '.npy'), out_code)
    np.save(os.path.join(save_path, 'generate' + prefix + '.npy'), out_poses)
    return out_poses, out_code


def visualizeCodeAndWrite(code_path=None, save_path="./output/",
                          prefix=None, pipeline_path="./dataset/data_pipe_60_rotation.sav",
                          generateGT=True, code_source=None, vis=True):
    bvh_path = './dataset/BEAT0909/Motion/1_wayne_0_103_110.bvh'
    model_path = config.VQVAE_model_path
    # bvh_path = "/mnt/nfs7/y50021900/My/data/Trinity_Speech-Gesture_I/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh"
    # model_path = '/mnt/nfs7/y50021900/My/codebook/Trinity_output_60fps_rotation/train_codebook/' + "codebook_checkpoint_best.bin"
    # pipeline_path = '../process/resource/data_pipe_60.sav'
    save_path = os.path.join(save_path, prefix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if generateGT:
        print('process bvh...')
        if not os.path.exists(os.path.join(save_path, 'rotation' + 'GT' + '.npy')):
            poses, _ = process_bvh(bvh_path, modetype='rotation')
            np.save(os.path.join(save_path, 'rotation' + 'GT' + '.npy'), poses)
        else:
            print('npy already exists!')
            poses = np.load(os.path.join(save_path, 'rotation' + 'GT' + '.npy'))
        make_bvh_GENEA2020_BT(save_path, filename_prefix='GT', poses=poses, smoothing=False, pipeline_path=pipeline_path)
    print('inference code and rotation pose...')

    if code_source is None:
        code_source = np.load(code_path)['knn_pred']

    out_poses, out_code = visualize_code(config, model_path, save_path, prefix, code_source, normalize=True)
    print('rotation npy to bvh...')
    make_bvh_GENEA2020_BT(save_path, prefix, out_poses, smoothing=False, pipeline_path=pipeline_path)

    if vis:
        print('bvh to position npy...')
        bvh_path_generated = os.path.join(save_path, prefix + '_generated.bvh')
        bvh_to_npy(bvh_path_generated, save_path)
        print('visualize code...')
        npy_generated = np.load(os.path.join(save_path, prefix + '_generated.npy'))
        out_video = os.path.join(save_path, prefix + '_generated.mp4')
        visualize(npy_generated.reshape((npy_generated.shape[0], -1, 3)), out_video, out_code.flatten(), 'upper')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    # print("args", args)

    with open(args.configs) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    mydevice = torch.device('cuda:' + config.gpu)
    config.no_cuda = config.gpu

    print(config.no_cuda)
    # model = VQVAE(config.VQVAE, 15 * 9)

    visualizeCodeAndWrite(code_path=config.code_path, prefix=config.prefix, generateGT=False, save_path=config.save_path)
