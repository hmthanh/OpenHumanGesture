import argparse


def get_configs():
    parser = argparse.ArgumentParser(description='Codebook')
    parser.add_argument('-d', '--train_database', help="Path to training database.", type=str, default="./dataset/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_txt.npz")
    parser.add_argument('-c', '--train_codebook', help="Path to training database.", type=str, default="./dataset/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_code.npz")
    parser.add_argument('-w', '--train_wavlm', help="Path to training database.", type=str, default="./dataset/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_WavLM.npz")
    parser.add_argument('-wvq', '--train_wavvq', help="Path to training database.", type=str, default="./dataset/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_WavVQ.npz")
    parser.add_argument('-s', '--codebook_signature', help="Path to training database.", type=str, default="./dataset/BEAT/BEAT_output_60fps_rotation/code.npz")
    parser.add_argument('-e', '--test_data', help="Path to test data.", type=str, default="./dataset/BEAT/speaker_10_state_0/speaker_10_state_0_test_240_txt.npz")
    parser.add_argument('-tw', '--test_wavlm', help="Path to training database.", type=str, default="./dataset/BEAT/speaker_10_state_0/speaker_10_state_0_test_240_WavLM.npz")
    parser.add_argument('-twvq', '--test_wavvq', help="Path to training database.", type=str, default="./dataset/Example1/ZeroEGGS_cut/wavvq_240.npz")
    parser.add_argument('-om', '--out_knn_filename', help="Output filename of the k-NN searched motion.", type=str, default="./codebook/Speech2GestureMatching/output/result.npz")
    parser.add_argument('-ov', '--out_video_path', help="Output path of the video.", type=str, default="./codebook/Speech2GestureMatching/output/output_video_folder/")
    parser.add_argument('-k', '--desired_k', help="The desired k-value for the k-NN (starts from 0).", type=int, default=0)
    parser.add_argument('-f', '--fake', type=bool, default=False)
    parser.add_argument('-of', '--out_fake_knn_filename', help="Output filename of the k-NN searched motion.", type=str, default="/path/to/knn_pred.npz")
    parser.add_argument('--max_frames', type=int, default=0)
    args = parser.parse_args()
    return args
