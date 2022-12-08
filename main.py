import sys
import argparse
from distutils.util import strtobool

from load_data import *
from prepare_data import *
from train import *

import warnings
warnings.filterwarnings('ignore')

def main(config):
    train = config.train
    emotion_type = config.emotion
    frame_skip = 2 # We skip every 2 frames to compute spotting dataset, for time-saving

    # Determine Emotion
    label_dict, emotion_dict = determine_emotion(emotion_type)

    ##### Recognition #####
    # Load Data
    print('\n ------ Loading Processed Recognition Data ------')
    final_dataset, final_subjects, final_videos, final_emotions = load_processed_recog_data()
    
    # Prepare Data
    print('\n ------ Preparing Recognition Data ------')
    X_recog, Y1_recog, groupsLabel_recog = prepare_recog_data(final_dataset, final_subjects, final_emotions, label_dict, emotion_dict, emotion_type)

    ##### Spotting #####
    # Load Data
    print('\n ------ Loading Processed Spotting Data ------')
    code_final = get_annotation()
    dataset, final_subjects, final_videos, final_exp, final_samples, final_emotions = load_processed_spotting_data(code_final, emotion_dict)

    # Prepare Data
    print('\n ------ Preparing Spotting Data ------')
    pseudo_y, pseudo_y1 = pseudo_labeling(frame_skip, dataset, final_samples, final_emotions, final_exp, label_dict, emotion_type)
    X_spot, Y_spot, Y1_spot, groupsLabel_spot = prepare_spot_data(dataset, final_subjects, final_samples, pseudo_y, pseudo_y1)

    # Setting paramaters
    model_name = 'RGBD'
    batch_size = 1024
    loss_lambda = 0.9
    steps = 1000
    lr = 0.0001
    attempt = 1
    ratio = 1
    dif_threshold = 0.42
    micro_threshold = 0.576
    macro_threshold = 0.576
    video_threshold = 0.41
    show = False
    emotion_type += 1 # Bcoz of neutral emotion

    # Training & Evaluation
    print('\n ------ 3D-MEAN Training and Evaluation ------')
    train_model(train, X_spot, Y_spot, Y1_spot, X_recog, Y1_recog, groupsLabel_spot, groupsLabel_recog, dataset, final_subjects, final_exp, final_samples, final_videos, final_emotions, label_dict, frame_skip, emotion_type, steps, lr, batch_size, model_name, attempt, ratio, micro_threshold, macro_threshold, dif_threshold, video_threshold, loss_lambda, show)

    print('\n ------ Completed ------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # input parameters
    parser.add_argument('--train', type=strtobool, default=True) # True or False
    parser.add_argument('--emotion', type=int, default=4) # 4emo or 7emo

    config = parser.parse_args()
    main(config)