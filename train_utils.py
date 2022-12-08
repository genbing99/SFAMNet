
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import Counter
import random
from sklearn.metrics import confusion_matrix
from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d
random.seed(1)

def confusionMatrix(gt, pred, show=False):
    TN_recog, FP_recog, FN_recog, TP_recog = confusion_matrix(gt, pred).ravel()
    f1_score = (2*TP_recog) / (2*TP_recog + FP_recog + FN_recog)
    num_samples = len([x for x in gt if x==1])
    average_recall = TP_recog / (TP_recog + FN_recog)
    average_precision = TP_recog / (TP_recog + FP_recog)
    return f1_score, average_recall, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, average_precision, average_recall

def recognition_evaluation(final_gt, final_pred, label_dict, show=False):
    #Display recognition result
    precision_list = []
    recall_list = []
    f1_list = []
    ar_list = []
    TP_all = 0
    FP_all = 0
    FN_all = 0
    TN_all = 0
    try:
        for emotion, emotion_index in label_dict.items():
            if emotion == 'neutral': #  Ignore the neutral emotion
                continue
            gt_recog = [1 if x==emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x==emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, precision_recog, recall_recog = confusionMatrix(gt_recog, pred_recog, show)
                if(show):
                    print(emotion.title(), 'Emotion:')
                    print('TP:', TP_recog, '| FP:', FP_recog, '| FN:', FN_recog, '| TN:', TN_recog)
#                     print('Total Samples:', num_samples, '| F1-score:', round(f1_recog, 4), '| Average Recall:', round(recall_recog, 4), '| Average Precision:', round(precision_recog, 4))
                TP_all += TP_recog
                FP_all += FP_recog
                FN_all += FN_recog
                TN_all += TN_recog
                precision_list.append(precision_recog)
                recall_list.append(recall_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
#                 print('Recognition evaluation error:', e)
                pass
        precision_list = [0 if np.isnan(x) else x for x in precision_list]
        recall_list = [0 if np.isnan(x) else x for x in recall_list]
        precision_all = np.mean(precision_list)
        recall_all = np.mean(recall_list)
        f1_all = (2 * precision_all * recall_all) / (precision_all + recall_all)
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        if (show):
            print('------ After adding ------')
            print('TP:', TP_all, 'FP:', FP_all, 'FN:', FN_all, 'TN:', TN_all)
            print('Precision:', round(precision_all, 4), 'Recall:', round(recall_all, 4))
        return np.nan_to_num(UF1), np.nan_to_num(UAR), np.nan_to_num(f1_all) # Return 0 if nan
    except:
        return 0, 0, 0

def calF1(TP, FP, FN):
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = (2 * precision * recall) / (precision + recall)
    except:
        precision = recall = F1_score = 0
    return precision, recall, F1_score
    
def evaluation(cur_exp, metric): #Get TP, FP, FN for final evaluation
    TP = int(sum(metric.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP = int(sum(metric.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN = cur_exp - TP
    precision, recall, F1_score = calF1(TP, FP, FN)
    return TP, FP, FN, F1_score, precision, recall

def detectInterval(score_plot_agg, peak, left_dis, right_dis, threshold): # dis = distance to left and right of the peak
    start = peak
    best_diff = 0
    for left_index in range(peak-left_dis,peak+1):
        if left_index >= 0:
            diff = abs(score_plot_agg[peak] - score_plot_agg[left_index])
            if diff > best_diff and score_plot_agg[left_index] > threshold:
                start = left_index
                best_diff = diff
    end = peak
    best_diff = 0
    for right_index in range(peak,peak+right_dis+1):
        if right_index < len(score_plot_agg):
            diff = abs(score_plot_agg[peak] - score_plot_agg[right_index])
            if diff > best_diff and score_plot_agg[right_index] > threshold:
                end = right_index
                best_diff = diff
    return start, peak, end

# For score aggregation, to smooth the spotting confidence score
def smooth(y, box_pts):
    y = [each_y for each_y in y]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def spot_then_recognize(result, result1, dataset, final_subjects, final_videos, final_samples, final_exp, final_emotions, label_dict, frame_skip, subject_count, micro_threshold, macro_threshold, dif_threshold, video_threshold, show=False):
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # parameter settings
    micro_min = int(15/frame_skip)
    micro_max = int(32/frame_skip)
    macro_min = int(25/frame_skip)
    macro_max = int(90/frame_skip)
    micro_left_dis = int(15/frame_skip)
    micro_right_dis = int(32/frame_skip)
    macro_left_dis = int(50/frame_skip)
    macro_right_dis = int(50/frame_skip)
    k_micro = int(15/frame_skip) # Modified
    k_macro = int(20/frame_skip)
    micro_phase_dif = 0.017
    macro_phase_dif = 0
    ignore_frame = 2
        
    prev = 0
    metric_micro = MeanAveragePrecision2d(num_classes=1)
    metric_macro = MeanAveragePrecision2d(num_classes=1)
    videos = [ele for ele in final_videos for ele in ele]
    
    for videoIndex, video in enumerate(final_samples[subject_count]):
        
        preds_micro = [] # [xmin, ymin, xmax, ymax, class_id, confidence, apex, pred_emotion] 
        preds_macro = []
        gt_micro = [] # [xmin, ymin, xmax, ymax, class_id, difficult, apex, gt_emotion]
        gt_macro = []
        micro_detected = []
        macro_detected = []
        countVideo = len([video for subject in final_samples[:subject_count] for video in subject])
        score_plot = np.array(result[prev:prev+len(dataset[countVideo+videoIndex])]) #Get related frames to each video
        pred_emotion_video = np.array(result1[prev:prev+len(dataset[countVideo+videoIndex])])
        
        score_plot_micro = smooth(score_plot, k_micro*2)
        score_plot_macro = smooth(score_plot, k_macro*2)
        
        #Plot the result to see the peaks
        if show:
            print('\nSubject:', final_subjects[subject_count], subject_count, 'Video:', videos[countVideo+videoIndex], countVideo+videoIndex)
            plt.figure(figsize=(15,3))
            plt.plot(score_plot_micro[ignore_frame:-ignore_frame], color=color_list[0]) 
            plt.plot(score_plot_macro[ignore_frame:-ignore_frame], color=color_list[3]) 
            plt.xlabel('Frame')
            plt.ylabel('Score')
            
        # Detect Micro
        peaks_micro, _ = find_peaks(score_plot_micro[k_micro:-k_micro], height=micro_threshold, distance=int((micro_left_dis+micro_right_dis)/2))
        for peak in peaks_micro: 
            peak = peak + k_micro - 1
            start, peak, end = detectInterval(score_plot_micro, peak, micro_left_dis, micro_right_dis, micro_threshold)
            if end-start >= micro_min and end-start <= micro_max:
                if score_plot_micro[start:end+1].mean() < 0.725:
                    if score_plot_micro[peak] - score_plot_micro[start] > micro_phase_dif and score_plot_micro[peak] - score_plot_micro[end] > micro_phase_dif:
                        micro_detected.append([start, peak, end])

        # Detect Macro
        peaks_macro, _ = find_peaks(score_plot_macro[k_macro:-k_macro], height=macro_threshold, distance=int((macro_left_dis+macro_right_dis)/2))
        for peak in peaks_macro:
            peak = peak + k_macro
            start, peak, end = detectInterval(score_plot_macro, peak, macro_left_dis, macro_right_dis, macro_threshold)
            if end-start < macro_min:
                start = max(start - int(macro_left_dis/3), 0)
                end = min(end + int(macro_left_dis/3), len(score_plot_macro)-1)
            if end-start >= macro_min and end-start <= macro_max:
                if score_plot_macro[peak] - score_plot_macro[start] > macro_phase_dif and score_plot_macro[peak] - score_plot_macro[end] > macro_phase_dif:
                    macro_detected.append([start, peak, end])
                    
        if score_plot_micro.mean() < video_threshold:
            for micro_phase in micro_detected:
                # Get pred emotion, Using spotted onset until apex
                pred_emotion_list = pred_emotion_video[max(0, micro_phase[0]) : min(len(score_plot_micro), micro_phase[1]+1)]
                most_common_emotion, _ = Counter(pred_emotion_list).most_common(1)[0]
                preds_micro.append([micro_phase[0]*frame_skip, 0, micro_phase[2]*frame_skip, 0, 0, 0, 0, most_common_emotion])
            for macro_phase in macro_detected:
                # Get pred emotion, Using spotted onset until apex
                pred_emotion_list = pred_emotion_video[max(0, macro_phase[0]) : min(len(score_plot_macro), macro_phase[1]+1)]
                most_common_emotion, _ = Counter(pred_emotion_list).most_common(1)[0]
                preds_macro.append([macro_phase[0]*frame_skip, 0, macro_phase[2]*frame_skip, 0, 0, 0, 0, most_common_emotion])

        gt_micro_list = []
        gt_macro_list = []
        for sampleIndex, samples in enumerate(video):
            if final_exp[subject_count][videoIndex][sampleIndex] == 'micro':
                offset = samples[2]
                if samples[2] - samples[0] > 30: # Cas3 Github: onset-apex-(apex+(apex-onset)) https://github.com/jingtingEmmaLi/CAS-ME-3
                    offset = samples[1] + (samples[1] - samples[0])
                if samples[0] == samples[1] == offset:
                    continue
                gt_micro.append([samples[0], 0, offset, 0, 0, 0, 0, 0, int(label_dict[final_emotions[subject_count][videoIndex][sampleIndex]])])
                gt_micro_list.append([samples[0], samples[1], offset])
            else:
                gt_macro.append([samples[0], 0, samples[2], 0, 0, 0, 0, 0, -1]) # emotion is +1 because cas3 dataset does not provide emotion label for macro-exp
                gt_macro_list.append([samples[0], samples[1], samples[2]])
            if show:
                if final_exp[subject_count][videoIndex][sampleIndex] == 'micro':
                    plt.axvline(x=int(samples[0]/frame_skip-ignore_frame), color=color_list[0])
                    plt.axvline(x=int(samples[2]/frame_skip-ignore_frame), color=color_list[0])
                else:
                    plt.axvline(x=int(samples[0]/frame_skip-ignore_frame), color=color_list[3])
                    plt.axvline(x=int(samples[2]/frame_skip-ignore_frame), color=color_list[3])
        if show:
            print('Micro video mean:', round(score_plot_micro.mean(), 4))
            if score_plot_micro.mean() >= video_threshold:
                print('Not included')
            print('Micro Before:', len(peaks_micro), 'After:', len(preds_micro))
            print('GT:', gt_micro_list)
            print('Preds:', (np.array(micro_detected) * frame_skip).tolist())
            print('Macro Before:', len(peaks_macro), 'After:', len(preds_macro))
            print('GT:', gt_macro_list)
            print('Preds:', (np.array(macro_detected) * frame_skip).tolist())
            plt.axhline(y=micro_threshold, color=color_list[0])
            plt.axhline(y=macro_threshold, color=color_list[3])
            
            # Change x ticks values
            x_ticks = []
            for item in plt.xticks()[0]:
                x_ticks.append(item)
            x_ticks = x_ticks[1:-1]
            x_ticks_modified = x_ticks.copy()
            for index, item in enumerate(x_ticks_modified):
                x_ticks_modified[index] = int(x_ticks_modified[index] * frame_skip)
            plt.xticks(x_ticks, x_ticks_modified)
        
        #Occurs when no peak is detected, simply give a value to pass the exception in mean_average_precision
        if len(preds_micro) == 0:
            preds_micro.append([0, 0, 0, 0, 0, -1, 0, -1]) # -1 to bypass the count of additional fp
        if len(preds_macro) == 0:
            preds_macro.append([0, 0, 0, 0, 0, -1, 0, -1])
        if len(gt_micro) == 0:
            gt_micro.append([-1, 0, -1, 0, 0, 0, 0, 0, -1])
        if len(gt_macro) == 0:
            gt_macro.append([-1, 0, -1, 0, 0, 0, 0, 0, -1])
            
        metric_micro.add(np.array(preds_micro), np.array(gt_micro)) # IoU = 0.5
        metric_macro.add(np.array(preds_macro), np.array(gt_macro)) # IoU = 0.5 
                
        if show:
            print('Micro TP:', int(sum(metric_micro.value(iou_thresholds=0.5)[0.5][0]['tp'])), '| Macro TP:', int(sum(metric_macro.value(iou_thresholds=0.5)[0.5][0]['tp'])), )
            plt.show()
        
        prev += len(dataset[countVideo+videoIndex])
    return metric_micro, metric_macro

def downSampling(Y_spot, Y1_recog):
    #Downsampling non expression samples to make ratio expression:non-expression 1:ratio
    ratio = 1
    rem_count = 0
    for key, value in Counter(Y_spot).items():
        if key == 1:
            rem_count += value
    rem_count += len(Y1_recog)
    rem_count = rem_count * ratio

    #Randomly remove non expression samples (With label 0) from dataset
    rem_index = random.sample([index for index, i in enumerate(Y_spot) if i==0], rem_count) 
    rem_index += (index for index, i in enumerate(Y_spot) if i==1)
    rem_index.sort()
    
    # Simply return 50 index
    if len(rem_index) == 0:
        print('No index selected')
        rem_index = [i for i in range(50)]
    return rem_index

def history_plot_pytorch(history):
    #Loss vs Epochs
    f, ax = plt.subplots(1,4,figsize=(25,4)) 
    ax[0].plot(history['loss'])
    ax[0].plot(history['val_loss'])
    ax[0].set_title('loss/epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['train_loss','val_loss'], loc='upper left')
    #Accuracy vs Epochs
    ax[1].plot(history['acc'])
    ax[1].plot(history['val_acc'])
    ax[1].set_title('accuracy/epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['acc','val_acc'], loc='upper left')
    #Spot Loss vs Epochs
    ax[2].plot(history['spot_loss'])
    ax[2].plot(history['val_spot_loss'])
    ax[2].set_title('spot loss/epochs')
    ax[2].set_ylabel('Spot Loss')
    ax[2].set_xlabel('Epoch')
    ax[2].legend(['spot_loss','val_spot_loss'], loc='upper left')
    #Recog Loss vs Epochs
    ax[3].plot(history['recog_loss'])
    ax[3].plot(history['val_recog_loss'])
    ax[3].set_title('recog loss/epochs')
    ax[3].set_ylabel('Recog Loss')
    ax[3].set_xlabel('Epoch')
    ax[3].legend(['recog_loss','val_recog_loss'], loc='upper left')
    plt.show()