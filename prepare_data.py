import pandas as pd
import numpy as np
from collections import Counter
from tensorflow.keras.utils import to_categorical
import copy

def determine_emotion(emotion_type):
    # 7 emotions
    if emotion_type == 7:
        label_dict = { 'disgust' : 0, 'surprise' : 1, 'others' : 2, 'fear' : 3, 'anger' : 4, 'sad' : 5, 'happy' : 6, 'neutral' : 7 }
        emotion_dict = { 'disgust' : 'disgust', 'surprise' : 'surprise', 'others' : 'others', 'fear' : 'fear', 'anger' : 'anger', 'sad' : 'sad', 'happy' : 'happy' }

    # 4 emotions
    if emotion_type == 4:
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3, 'neutral' : 4}
        emotion_dict = { 'happy': 'positive', 'disgust': 'negative', 'fear': 'negative', 'anger': 'negative', 'sad': 'negative', 'surprise': 'surprise', 'others': 'others' }

    return label_dict, emotion_dict

def prepare_recog_data(final_dataset, final_subjects, final_emotions, label_dict, emotion_dict, emotion_type):
    #To split the dataset by subjects
    groupsLabel_recog = final_subjects.copy()
    X_recog = final_dataset
    Y1_recog = final_emotions

    # Convert to 4 emotions
    if emotion_type == 4:
        Y1_recog = list(pd.Series(Y1_recog).map(emotion_dict))
        
    print(emotion_type, 'Emotions:', Counter(Y1_recog))
    emotion_type += 1 # To include neutral emotion
    Y1_recog = [label_dict[ele] for ele in Y1_recog]
    Y1_recog = to_categorical(Y1_recog)
    print('Total X :', len(X_recog))
    print('Total y :', len(Y1_recog))
    
    return X_recog, Y1_recog, groupsLabel_recog

def pseudo_labeling(frame_skip, dataset, final_samples, final_emotions, final_exp, label_dict, emotion_type):
    # Since optical flow compute for every 5 frames only
    final_samples_labeling = copy.deepcopy(final_samples)
    for subject_index, subject in enumerate(final_samples_labeling):
        for video_index, video in enumerate(subject):
            for sample_index, sample in enumerate(video):
                for phase_index, phase in enumerate(sample):
                    final_samples_labeling[subject_index][video_index][sample_index][phase_index] = int(phase/frame_skip)

    #### Pseudo-labeling
    pseudo_y = []
    pseudo_y1 = []
    video_count = 0 

    for subject_index, subject in enumerate(final_samples_labeling):
        for video_index, video in enumerate(subject):
            samples_arr = []
            pseudo_y_each = [0]*(len(dataset[video_count]))
            pseudo_y1_each = [emotion_type-1]*(len(dataset[video_count]))
            for sample_index, sample in enumerate(video):
                onset = sample[0]
                apex = sample[1]
                offset = sample[2]
                if final_exp[subject_index][video_index][sample_index] == 'micro':
                    start = int((onset+apex)/2)
                    end = apex
                    for frame_index, frame in enumerate(range(start, end+1)):
                        if frame < len(pseudo_y_each):
                            pseudo_y_each[frame] = 1 # Hard label
                            pseudo_y1_each[frame] = label_dict[final_emotions[subject_index][video_index][sample_index]]
            pseudo_y1.append(pseudo_y1_each)
            pseudo_y.append(pseudo_y_each)
            video_count+=1
            
    print('Total video:', len(pseudo_y))
    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    pseudo_y1 = [y1 for x in pseudo_y1 for y1 in x]
    print('Total frames:', len(pseudo_y))
    print('Distribution hard label:', Counter(pseudo_y))
    print('Emotion label:', Counter(pseudo_y1))
    # print('Distribution:', Counter(pseudo_y1))

    return pseudo_y, pseudo_y1

def prepare_spot_data(dataset, final_subjects, final_samples, pseudo_y, pseudo_y1):
    #To split the dataset by subjects
    Y_spot = np.array(pseudo_y)
    Y1_spot = to_categorical(pseudo_y1)
    videos_len = []
    groupsLabel_spot = Y_spot.copy()
    prevIndex = 0
    countVideos = 0

    #Get total frames of each video
    for video_index in range(len(dataset)):
        videos_len.append(len(dataset[video_index]))

    # print('Frame Index for each subject:-')
    for subject_index in range(len(final_samples)):
        countVideos += len(final_samples[subject_index])
        index = sum(videos_len[:countVideos])
        groupsLabel_spot[prevIndex:index] = final_subjects[subject_index]
        # print('Subject', final_subjects[subject_index], ':', prevIndex, '->', index)
        prevIndex = index

    X_spot = []
    for video_index, video in enumerate(dataset):
        X_spot.append(video)
    X_spot = [frame for video in X_spot for frame in video]
    print('\nTotal X:', len(X_spot), ', Total Y:', len(Y_spot), ', Total Y1:', len(Y1_spot))

    return X_spot, Y_spot, Y1_spot, groupsLabel_spot