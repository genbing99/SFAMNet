import pickle
import numpy as np
import pandas as pd
from collections import Counter

def load_processed_recog_data():
    dataset_all = pickle.load( open( "processed_data/CASME_cube_recog_rgbd-flow.pkl", "rb" ) )

    final_dataset = dataset_all[0]
    subjects = dataset_all[1]
    final_videos = dataset_all[2]
    final_emotions = dataset_all[3]
        
    final_subjects = [int(subject[5:]) for subject in subjects] # Get subject number only
    for index, emotion in enumerate(final_emotions):
        final_emotions[index] = final_emotions[index].lower()

    print('Total subjects: ', len(np.unique(final_subjects)))
    print('Total videos:', len(final_videos))
    print('Total emotions in each class:', Counter(final_emotions))

    return final_dataset, final_subjects, final_videos, final_emotions

def get_annotation():
    # Extracted from https://en.wikipedia.org/wiki/Facial_Action_Coding_System
    action_unit_dict = {
        0: 'Neutral face', 1: 'Inner brow raiser', 2: 'Outer brow raiser', 4: 'Brow lowerer', 5: 'Upper lid raiser',
        6: 'Cheek raiser', 7: 'Lid tightener',  8: 'Lips toward each other', 9: 'Nose wrinkler', 10: 'Upper lip raiser',
        11: 'Nasolabial deepener', 12: 'Lip corner puller', 13: 'Sharp lip puller', 14: 'Dimpler', 15: 'Lip corner depressor',
        16: 'Lower lip depressor',17: 'Chin raiser', 18: 'Lip pucker', 19: 'Tongue show', 20: 'Lip stretcher',
        21: 'Neck tightener', 22: 'Lip funneler', 23: 'Lip tightener', 24: 'Lip pressor', 25: 'Lips part',
        26: 'Jaw drop', 27: 'Mouth stretch', 28: 'Lip suck', 29: 'Jaw thrust', 38: 'Nostril dilator', 39: 'Nostril compressor', 
        41: 'Lid droop', 42: 'Slit', 43: 'Eyes Closed', 44: 'Squint', 45: 'Blink', 46: 'Wink'
    }
    xl = pd.ExcelFile('annotation/CAS(ME)3_part_A_v2.xls') #Specify directory of excel file
    colsName = ['subject', 'video', 'onset', 'apex', 'offset', 'au', 'type', 'objective class']
    code_final1 = xl.parse(xl.sheet_names[0], names=colsName) #Get data

    xl = pd.ExcelFile('annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx', engine="openpyxl") #Specify directory of excel file
    colsName = ['subject', 'video', 'onset', 'apex', 'offset', 'au', 'objective class', 'emotion']
    code_final2 = xl.parse(xl.sheet_names[0], names=colsName) #Get data

    code_final = pd.merge(code_final1, code_final2, on=['subject', 'video', 'onset', 'apex', 'offset'], how='outer')
    code_final['au'] = code_final['au_x'].fillna('unknown')
    code_final['emotion'].fillna('unknown', inplace=True)

    # Get subject number
    subject_num = []
    for subject in code_final['subject']:
        subject_num.append(int(subject.split('.')[1]))
    code_final['subject_num'] = subject_num

    # Get AU
    au_name = []
    for au in code_final['au']:
        au_name_each = []
        for au_each in au.split('+'):
            if 'L' in au_each.upper():
                au_name_each.append('Left ' + action_unit_dict[int(au_each[1:])])
            elif 'R' in au_each.upper():
                au_name_each.append('Right ' + action_unit_dict[int(au_each[1:])])
            elif 'T' in au_each.upper():
                au_name_each.append('Top ' + action_unit_dict[int(au_each[1:])])
            elif 'B' in au_each.upper():
                au_name_each.append('Bottom ' + action_unit_dict[int(au_each[1:])])
            elif au_each == 'unknown':
                au_name_each.append('Unknown')
            else:
                au_name_each.append(action_unit_dict[int(au_each)])
        au_name.append(au_name_each)
    code_final['au_name'] = au_name

    code_final.sort_values(by=['subject_num', 'video'], inplace=True)
    
    # print('Data Columns:', code_final.columns) #Final data column

    return code_final

def load_processed_spotting_data(code_final, emotion_dict):
    dataset_all = pickle.load( open( "processed_data/CASME_cube_spot_rgbd-flow.pkl", "rb" ) )

    dataset = dataset_all[0]
    subjects = dataset_all[1]
    videos = dataset_all[2]

    final_subjects = []
    final_videos = []
    final_exp = []
    final_samples = []
    final_emotions = []
    micro = 0
    macro = 0
    ground_truth = []
    for sub_video_each_index, sub_vid_each in enumerate(videos):
        ground_truth.append([])
        final_exp.append([])
        final_samples.append([])
        final_emotions.append([])
        for videoIndex, videoCode in enumerate(sub_vid_each):
            on_off = []
            exp_subject = []
            emotion_subject = []
            for i, row in code_final.loc[(code_final['subject'] == subjects[sub_video_each_index]) & (code_final['video'] == videoCode)].iterrows():
                if (row['type']=='Micro-expression'): #Micro-expression or macro-expression
                    micro += 1
                    exp_subject.append('micro')
                    emotion_subject.append(emotion_dict[row['emotion'].lower()])
                else:
                    macro += 1
                    exp_subject.append('macro')
                    emotion_subject.append('neutral')
                on_off.append([int(row['onset']-1), int(row['apex']-1), int(row['offset']-1)])
            final_samples[-1].append(on_off) 
            final_exp[-1].append(exp_subject)
            final_emotions[-1].append(emotion_subject)
            
    final_subjects = [int(subject[5:]) for subject in subjects] # Get subject number only
    final_videos = videos

    print('Final Ground Truth Data')
    print('Subjects Name', final_subjects)
    # print('Videos Name: ', final_videos)
    # print('Samples [Onset, Offset]: ', final_samples)
    # print('Expression Type:', final_exp)
    # print('Emotions:', final_emotions)
    print('Total Micro:', micro, 'Macro:', macro)

    return dataset, final_subjects, final_videos, final_exp, final_samples, final_emotions
