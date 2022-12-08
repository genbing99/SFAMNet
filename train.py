import os
import time
from sklearn.model_selection import LeaveOneGroupOut
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
import torch.nn as nn
from sklearn.utils import class_weight

from train_utils import *
from dataloader import *
from network import *

def train_model(train, X_spot, Y_spot, Y1_spot, X_recog, Y1_recog, groupsLabel_spot, groupsLabel_recog, dataset, final_subjects, final_exp, final_samples, final_videos, final_emotions, label_dict, frame_skip, emotion_type, steps, lr, batch_size, model_name, attempt, ratio, micro_threshold, macro_threshold, dif_threshold, video_threshold, loss_lambda, show):
    
    # Create model directory
    if train:
        os.makedirs("save_models/%s_STR_%semo/a%s" % (model_name, emotion_type-1, attempt), exist_ok=True)

    start = time.time()
    loso = LeaveOneGroupOut()
    subject_count = 0
    is_cuda = torch.cuda.is_available()
    transform = None
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Spot
    spot_train_index = []
    spot_test_index = []
    metric_fn_micro = MeanAveragePrecision2d(num_classes=1)
    metric_fn_macro = MeanAveragePrecision2d(num_classes=1)
    metric_micro_total = MeanAveragePrecision2d(num_classes=1)
    metric_macro_total = MeanAveragePrecision2d(num_classes=1)
    metric_overall_total = MeanAveragePrecision2d(num_classes=1)
    TP_micro_total = FP_micro_total = FN_micro_total = 0
    TP_macro_total = FP_macro_total = FN_macro_total = 0
    TP_overall_total = FP_overall_total = FN_overall_total = 0
    result_final = []
    result1_final = []
    # LOSO
    for train_index, test_index in loso.split(X_spot, X_spot, groupsLabel_spot):
        spot_train_index.append(train_index)
        spot_test_index.append(test_index)

    # Recognition
    recog_train_index = []
    recog_test_index = []
    cur_gt = []
    cur_pred = []
    # Spot-then-recognize
    str_pred_all = []
    str_gt_all = []
    # LOSO
    missing_recog_subject = list(set(np.unique(groupsLabel_spot)).difference(np.unique(groupsLabel_recog)))
    missing_recog_subject.sort()
    missing_index = 0
    for train_index, test_index in loso.split(X_recog, Y1_recog, groupsLabel_recog):
        if len(missing_recog_subject) > missing_index and groupsLabel_recog[test_index[0]] > missing_recog_subject[missing_index]:
            recog_train_index.append([i for i in range(len(Y1_recog))]) # Since this subject has no samples
            recog_test_index.append([])
            missing_index += 1
        recog_train_index.append(train_index)
        recog_test_index.append(test_index)
    recog_train_index.append([i for i in range(len(Y1_recog))]) # Since this subject has no samples
    recog_test_index.append([])

    # Training and Testing
    subjects_unique = sorted(np.unique(final_subjects))
    for subject_count in range(len(subjects_unique)): 

        cur_micro = 0
        cur_macro = 0
        for video_index, video_exp in enumerate(final_exp[subject_count]):
            for sample_index, sample_exp in enumerate(video_exp):
                samples = final_samples[subject_count][video_index][sample_index]
                onset = samples[0]
                apex = samples[1]
                offset = samples[2]
                if sample_exp == 'micro':
                    if offset - onset > 30:
                        offset = apex + (apex - onset)
                        if offset - onset != 0:
                            cur_micro += 1
                    else:
                        cur_micro += 1
                else:
                    cur_macro += 1

        # Use copy to ensure the original value is not modified
        X_spot_train, X_spot_test   = [X_spot[i] for i in spot_train_index[subject_count]], [X_spot[i] for i in spot_test_index[subject_count]]
        Y_spot_train, Y_spot_test   = [Y_spot[i] for i in spot_train_index[subject_count]], [Y_spot[i] for i in spot_test_index[subject_count]]
        Y1_spot_train, Y1_spot_test = [Y1_spot[i] for i in spot_train_index[subject_count]], [Y1_spot[i] for i in spot_test_index[subject_count]]

        X_recog_train, X_recog_test   = [X_recog[i] for i in recog_train_index[subject_count]], [X_recog[i] for i in recog_test_index[subject_count]] 
        Y1_recog_train, Y1_recog_test = [Y1_recog[i] for i in recog_train_index[subject_count]], [Y1_recog[i] for i in recog_test_index[subject_count]] 

        print('Subject : ' + str(subject_count+1), ', spNO.', subjects_unique[subject_count])

        # Create final dataset for training
        rem_index = downSampling(Y_spot_train, Y1_recog_train)
        # From spot dataset
        X_train_final = [X_spot_train[i] for i in rem_index]
        Y_train_final = [Y_spot_train[i] for i in rem_index]
        Y1_train_final = [argmax(Y1_spot_train[i]) for i in rem_index]
        # From recog dataset
        X_train_final.extend(X_recog_train)
        Y_train_final.extend([1 for i in range(len(X_recog_train))])
        Y1_train_final.extend(argmax(Y1_recog_train,-1).tolist())
        # Create final dataset for validation
        rem_index = downSampling(Y_spot_test, [])
        X_val_final = [X_spot_test[i] for i in rem_index]
        Y_val_final = [Y_spot_test[i] for i in rem_index]
        Y1_val_final = [argmax(Y1_spot_test[i]) for i in rem_index]
        # Create final dataset for testing
        X_test_final = X_spot_test
        Y_test_final = Y_spot_test
        Y1_test_final = argmax(Y1_spot_test,-1).tolist()

        print('Training   Dataset Labels, Spotting:', Counter(Y_train_final), ', Recognition:', Counter(Y1_train_final))
        print('Validation Dataset Labels, Spotting:', Counter(Y_val_final),   ', Recognition:', Counter(Y1_val_final))
        print('Testing    Dataset Labels, Spotting:', Counter(Y_test_final),  ', Recognition:', Counter(Y1_test_final))

        # Initialize training dataloader
        X_train_final = torch.Tensor(np.array(X_train_final)).permute(0,3,1,2)
        Y_train_final = torch.Tensor(np.array(Y_train_final))
        Y1_train_final= torch.Tensor(np.array(Y1_train_final)).type(torch.long)
        train_dl = DataLoader(
            OFFSTRDataset((X_train_final[:, 0][:, None, :], X_train_final[:, 1][:, None, :], X_train_final[:, 2][:, None, :], Y_train_final, Y1_train_final), transform=transform, train=True),
            batch_size=batch_size,
            shuffle=True,
        )
        # Initialize validation dataloader
        X_val_final = torch.Tensor(np.array(X_val_final)).permute(0,3,1,2)
        Y_val_final = torch.Tensor(np.array(Y_val_final))
        Y1_val_final = torch.Tensor(np.array(Y1_val_final)).type(torch.long)
        val_spot_dl = DataLoader(
            OFFSTRDataset((X_val_final[:, 0][:, None, :], X_val_final[:, 1][:, None, :], X_val_final[:, 2][:, None, :], Y_val_final, Y1_val_final), transform=transform, train=False),
            batch_size=batch_size,
            shuffle=False,
        )
        # Initialize testing dataloader
        X_test_final = torch.Tensor(np.array(X_test_final)).permute(0,3,1,2)
        Y_test_final = torch.Tensor(np.array(Y_test_final))
        Y1_test_final = torch.Tensor(np.array(Y1_test_final)).type(torch.long)
        test_spot_dl = DataLoader(
            OFFSTRDataset((X_test_final[:, 0][:, None, :], X_test_final[:, 1][:, None, :], X_test_final[:, 2][:, None, :], Y_test_final, Y1_test_final), transform=transform, train=False),
            batch_size=batch_size,
            shuffle=False,
        )

        if len(X_recog_test) > 0:
            X_recog_test_final = torch.Tensor(np.array(X_recog_test)).permute(0,3,1,2)
            Y_recog_test_final = torch.Tensor([1 for i in range(len(X_recog_test_final))]) # Useless
            Y1_recog_test_final = argmax(Y1_recog_test,-1).tolist()
            Y1_recog_test_final = torch.Tensor(np.array(Y1_recog_test_final)).type(torch.long)
            test_recog_dl = DataLoader(
                OFFSTRDataset((X_recog_test_final[:, 0][:, None, :], X_recog_test_final[:, 1][:, None, :], X_recog_test_final[:, 2][:, None, :], Y_recog_test_final, Y1_recog_test_final), transform=transform, train=False),
                batch_size=batch_size,
                shuffle=False,
            )

        # Loss function
        loss_fn_spot = nn.MSELoss()
        class_weights = class_weight.compute_class_weight('balanced', np.array([i for i in range(emotion_type-1)]), np.array(Y1_train_final[Y1_train_final != emotion_type-1]))
        class_weights = np.append(class_weights, 0) # Set neutral emotion class weight to zero
        class_weights = torch.tensor(class_weights,dtype=torch.float).cuda()
        loss_fn_recog = nn.CrossEntropyLoss(weight=class_weights,reduction='mean') 
        print('Class Weights:', class_weights)

        history = {} # Collects per-epoch loss and acc like Keras' fit().
        history['loss'] = []
        history['val_loss'] = []
        history['acc'] = []
        history['val_acc'] = []
        history['spot_loss'] = []
        history['val_spot_loss'] = []
        history['recog_loss'] = []
        history['val_recog_loss'] = []

        step = 0
        step_output = steps / 5
        train_loss = 0.0
        train_spot_loss = 0.0
        train_recog_loss = 0.0
        train_acc  = 0.0
        print('------Initializing Network-------') #To reset the model at every LOSO testing
        # model and optimizer
        model = Net_3D_MEAN(out_channels=emotion_type).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if train: # Test

            # Step as stop criteria
            while True:

                num_train_correct  = 0
                num_train_examples = 0

                # Training
                model.train()
                for batch in train_dl:

                    x1   = batch[0].to(device)
                    x2   = batch[1].to(device)
                    x3   = batch[2].to(device)
                    y    = batch[3].to(device)
                    y1   = batch[4].to(device)
                    optimizer.zero_grad()
                    yhat, yhat1 = model(x1,x2,x3)
                    yhat = yhat.view(-1)
                    yhat1 = yhat1.view(len(yhat1), emotion_type)

                    loss_spot = loss_fn_spot(yhat, y)
                    loss_recog = loss_fn_recog(yhat1, y1)
                    # Compute only non-neutral emotion
                    non_neutral = np.where(y1.cpu().numpy() != emotion_type-1)[0]
                    if len(non_neutral) > 0:
                        y1 = y1[non_neutral]
                        yhat1 = yhat1[non_neutral]
                        num_train_correct  += (torch.max(yhat1, 1)[1] == y1).sum().item()
                        num_train_examples += len(non_neutral)

                    loss = (loss_spot * loss_lambda) + (loss_recog * (1 - loss_lambda))
                    loss.backward()
                    optimizer.step()
                    train_acc   = num_train_correct / num_train_examples
                    train_recog_loss = loss_recog.data.item()

                    train_loss = loss.data.item()
                    train_spot_loss = loss_spot.data.item()

                    # Validation
                    model.eval()
                    val_loss = 0.0
                    val_spot_loss = 0.0    
                    for batch in val_spot_dl:
                        x1   = batch[0].to(device)
                        x2   = batch[1].to(device)
                        x3   = batch[2].to(device)
                        y    = batch[3].to(device)
                        y1   = batch[4].to(device)
                        yhat, yhat1 = model(x1,x2,x3)
                        yhat = yhat.view(-1)
                        yhat1 = yhat1.view(len(yhat1), emotion_type)

                        loss_spot = loss_fn_spot(yhat, y)
                        loss_recog = loss_fn_recog(yhat1, y1)
                        loss = (loss_spot * loss_lambda) + (loss_recog * (1 - loss_lambda))
                        val_loss += loss.data.item()
                        val_spot_loss += loss_spot.data.item()

                    val_loss = val_loss / len(val_spot_dl)
                    val_spot_loss = val_spot_loss / len(val_spot_dl)

                    # Testing recognition
                    model.eval()
                    val_recog_loss = 0.0
                    val_acc  = 0.0
                    num_val_correct  = 0
                    num_val_examples = 0 
                    pred = []
                    gt   = []
                    if len(X_recog_test) > 0:
                        for batch in test_recog_dl:
                            x1   = batch[0].to(device)
                            x2   = batch[1].to(device)
                            x3   = batch[2].to(device)
                            y    = batch[3].to(device)
                            y1   = batch[4].to(device)
                            yhat, yhat1 = model(x1,x2,x3)
                            yhat1 = yhat1.view(len(yhat1), emotion_type)
                            loss_recog = loss_fn_recog(yhat1, y1)
                            pred = torch.max(yhat1, 1)[1].tolist()
                            gt   = y1.tolist()
                            # Exclude neutral emotion
                            num_val_correct  += (torch.max(yhat1, 1)[1] == y1).sum().item()
                            num_val_examples += y1.size(0)
                            val_recog_loss += loss_recog.data.item()

                        val_acc  = num_val_correct / num_val_examples
                        val_recog_loss = val_recog_loss / len(test_recog_dl)

                        history['acc'].append(train_acc)
                        history['val_acc'].append(val_acc)
                        history['recog_loss'].append(train_recog_loss)
                        history['val_recog_loss'].append(val_recog_loss)

                    if step % step_output == 0:
                        # Testing spot-then-recognize
                        model.eval()
                        result_all = np.array([])
                        result1_all = np.array([])
                        for batch in test_spot_dl:
                            x1   = batch[0].to(device)
                            x2   = batch[1].to(device)
                            x3   = batch[2].to(device)
                            y    = batch[3].to(device)
                            y1   = batch[4].to(device)
                            yhat, yhat1 = model(x1,x2,x3)
                            yhat = yhat.view(-1)
                            result = yhat.cpu().data.numpy()
                            yhat1 = yhat1.view(len(yhat1), emotion_type)
                            result1 = torch.max(yhat1, 1)[1].tolist()
                            result_all = np.append(result_all, result)
                            result1_all = np.append(result1_all, result1)

                        # Spotting Evaluation
                        metric_micro, metric_macro = spot_then_recognize(result_all, result1_all, dataset, final_subjects, final_videos, final_samples, final_exp, final_emotions, label_dict, frame_skip, subject_count, micro_threshold, macro_threshold, dif_threshold, video_threshold, show=False)
                        TP_micro, FP_micro, FN_micro, F1_score_micro, precision_micro, recall_micro = evaluation(cur_micro, metric_micro)
                        TP_macro, FP_macro, FN_macro, F1_score_macro, precision_macro, recall_macro = evaluation(cur_macro, metric_macro)
                        TP_overall = TP_micro + TP_macro; FP_overall = FP_micro + FP_macro; FN_overall = FN_micro + FN_macro
                        precision_overall, recall_overall, F1_score_overall = calF1(TP_overall, FP_overall, FN_overall)
                        # Spot-then-recognize Evaluation
                        str_pred_subject = []
                        str_gt_subject = []
                        str_pred_list = metric_micro.get_pred()
                        str_gt_list = metric_micro.get_gt()
                        tp_micro_all = metric_micro.value(iou_thresholds=0.5)[0.5][0]['tp']
                        match_index_subject = metric_micro.value(iou_thresholds=0.5)[0.5][0]['match_index']
                        sample_count = 0
                        for video_index, video_val in enumerate(metric_micro.get_pred()):
                            for sample_index, sample_val in enumerate(video_val):
                                if tp_micro_all[sample_count] == 1:
                                    str_pred_subject.append(int(str_pred_list[video_index][sample_index][-1])) # Get emotion
                                    str_gt_subject.append(int(str_gt_list[video_index][match_index_subject[video_index][sample_index][0]][-1])) # Get emotion of the match index
                                sample_count += 1
                        UF1, UAR, F1_score = recognition_evaluation(str_gt_subject, str_pred_subject, label_dict, show=False)

                        # Display result
                        print('Train examples: %d, Test examples: %d' % (num_train_examples, num_val_examples))
                        print('Step %3d/%3d, train loss: %5.4f, train acc: %5.4f, test loss: %5.4f, test acc: %5.4f' % (step, steps, train_loss, train_acc, val_loss, val_acc))
                        print('Spotting Micro result: TP:%d FP:%d FN:%d F1_score:%5.4f' % (TP_micro, FP_micro, FN_micro, F1_score_micro))
                        print('Spotting Macro result: TP:%d FP:%d FN:%d F1_score:%5.4f' % (TP_macro, FP_macro, FN_macro, F1_score_macro))
                        print('Spotting Overall result: TP:%d FP:%d FN:%d F1_score:%5.4f' % (TP_overall, FP_overall, FN_overall, F1_score_overall))
                        print('Analysis Micro result: UF1:%5.4f, UAR:%5.4f, F1-score:%5.4f, STRS:%5.4f\n' % (UF1, UAR, F1_score, (F1_score_micro * F1_score)))

                    step += 1
                    if step == steps+1:
                        break

                    history['loss'].append(train_loss)
                    history['spot_loss'].append(train_spot_loss)
                    history['val_loss'].append(val_loss)
                    history['val_spot_loss'].append(val_spot_loss)

                if step == steps+1:
                    break

            # Save models
            torch.save(model.state_dict(), os.path.join("save_models/%s_STR_%semo/a%s/subject_%s.pkl" % (model_name, emotion_type-1, attempt, str(final_subjects[subject_count]))))

            # Plot training/val/loss graphs
            if show:
                history_plot_pytorch(history)

            result_final.append(result_all)
            result1_final.append(result1_all)
            
            _, _ = spot_then_recognize(result_all, result1_all, dataset, final_subjects, final_videos, final_samples, final_exp, final_emotions, label_dict, frame_skip, subject_count, micro_threshold, macro_threshold, dif_threshold, video_threshold, show=show)
            TP_micro_total += TP_micro; TP_macro_total += TP_macro; TP_overall_total += TP_overall
            FP_micro_total += FP_micro; FP_macro_total += FP_macro; FP_overall_total += FP_overall
            FN_micro_total += FN_micro; FN_macro_total += FN_macro; FN_overall_total += FN_overall
            precision_micro_total, recall_micro_total, F1_score_micro_total = calF1(TP_micro_total, FP_micro_total, FN_micro_total)
            precision_macro_total, recall_macro_total, F1_score_macro_total = calF1(TP_macro_total, FP_macro_total, FN_macro_total)
            precision_overall_total, recall_overall_total, F1_score_overall_total = calF1(TP_overall_total, FP_overall_total, FN_overall_total)
            for index in range(len(metric_micro.get_pred())):
                metric_micro_total.add(np.array(metric_micro.get_pred()[index]), np.array(metric_micro.get_gt()[index]))
                metric_overall_total.add(np.array(metric_micro.get_pred()[index]), np.array(metric_micro.get_gt()[index]))
            for index in range(len(metric_macro.get_pred())):
                metric_macro_total.add(np.array(metric_macro.get_pred()[index]), np.array(metric_macro.get_gt()[index]))
                metric_overall_total.add(np.array(metric_macro.get_pred()[index]), np.array(metric_macro.get_gt()[index]))
            AP_micro_total = metric_micro_total.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP']
            AP_macro_total = metric_macro_total.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP']
            AP_overall_total = metric_overall_total.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP']
        
        else: # Test

            if emotion_type == 5:
                model.load_state_dict(torch.load("pretrained_weights/4emo/subject_%s.pkl" % (str(final_subjects[subject_count]))))
            elif emotion_type == 8:
                model.load_state_dict(torch.load("pretrained_weights/7emo/subject_%s.pkl" % (str(final_subjects[subject_count]))))

            # Testing recognition
            model.eval()
            val_recog_loss = 0.0
            val_acc  = 0.0
            num_val_correct  = 0
            num_val_examples = 0 
            pred = []
            gt   = []
            if len(X_recog_test) > 0:
                for batch in test_recog_dl:
                    x1   = batch[0].to(device)
                    x2   = batch[1].to(device)
                    x3   = batch[2].to(device)
                    y    = batch[3].to(device)
                    y1   = batch[4].to(device)
                    yhat, yhat1 = model(x1,x2,x3)
                    yhat1 = yhat1.view(len(yhat1), emotion_type)
                    loss_recog = loss_fn_recog(yhat1, y1)
                    pred = torch.max(yhat1, 1)[1].tolist()
                    gt   = y1.tolist()
                    # Exclude neutral emotion
                    num_val_correct  += (torch.max(yhat1, 1)[1] == y1).sum().item()
                    num_val_examples += y1.size(0)
                    val_recog_loss += loss_recog.data.item()

            # Testing spot-then-recognize
            model.eval()
            result_all = np.array([])
            result1_all = np.array([])
            for batch in test_spot_dl:
                x1   = batch[0].to(device)
                x2   = batch[1].to(device)
                x3   = batch[2].to(device)
                y    = batch[3].to(device)
                y1   = batch[4].to(device)
                yhat, yhat1 = model(x1,x2,x3)
                yhat = yhat.view(-1)
                result = yhat.cpu().data.numpy()
                yhat1 = yhat1.view(len(yhat1), emotion_type)
                result1 = torch.max(yhat1, 1)[1].tolist()
                result_all = np.append(result_all, result)
                result1_all = np.append(result1_all, result1)

            result_final.append(result_all)
            result1_final.append(result1_all)
            # Spotting Evaluation
            metric_micro, metric_macro = spot_then_recognize(result_all, result1_all, dataset, final_subjects, final_videos, final_samples, final_exp, final_emotions, label_dict, frame_skip, subject_count, micro_threshold, macro_threshold, dif_threshold, video_threshold, show=show)
            TP_micro, FP_micro, FN_micro, F1_score_micro, precision_micro, recall_micro = evaluation(cur_micro, metric_micro)
            TP_macro, FP_macro, FN_macro, F1_score_macro, precision_macro, recall_macro = evaluation(cur_macro, metric_macro)
            TP_overall = TP_micro + TP_macro; FP_overall = FP_micro + FP_macro; FN_overall = FN_micro + FN_macro
            TP_micro_total += TP_micro; TP_macro_total += TP_macro; TP_overall_total += TP_overall
            FP_micro_total += FP_micro; FP_macro_total += FP_macro; FP_overall_total += FP_overall
            FN_micro_total += FN_micro; FN_macro_total += FN_macro; FN_overall_total += FN_overall
            precision_micro_total, recall_micro_total, F1_score_micro_total = calF1(TP_micro_total, FP_micro_total, FN_micro_total)
            precision_macro_total, recall_macro_total, F1_score_macro_total = calF1(TP_macro_total, FP_macro_total, FN_macro_total)
            precision_overall_total, recall_overall_total, F1_score_overall_total = calF1(TP_overall_total, FP_overall_total, FN_overall_total)
            for index in range(len(metric_micro.get_pred())):
                metric_micro_total.add(np.array(metric_micro.get_pred()[index]), np.array(metric_micro.get_gt()[index]))
                metric_overall_total.add(np.array(metric_micro.get_pred()[index]), np.array(metric_micro.get_gt()[index]))
            for index in range(len(metric_macro.get_pred())):
                metric_macro_total.add(np.array(metric_macro.get_pred()[index]), np.array(metric_macro.get_gt()[index]))
                metric_overall_total.add(np.array(metric_macro.get_pred()[index]), np.array(metric_macro.get_gt()[index]))
            AP_micro_total = metric_micro_total.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP']
            AP_macro_total = metric_macro_total.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP']
            AP_overall_total = metric_overall_total.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP']
            # Spot-then-recognize Evaluation
            str_pred_subject = []
            str_gt_subject = []
            str_pred_list = metric_micro.get_pred()
            str_gt_list = metric_micro.get_gt()
            tp_micro_all = metric_micro.value(iou_thresholds=0.5)[0.5][0]['tp']
            match_index_subject = metric_micro.value(iou_thresholds=0.5)[0.5][0]['match_index']
            sample_count = 0
            for video_index, video_val in enumerate(metric_micro.get_pred()):
                for sample_index, sample_val in enumerate(video_val):
                    if tp_micro_all[sample_count] == 1:
                        str_pred_subject.append(int(str_pred_list[video_index][sample_index][-1])) # Get emotion
                        str_gt_subject.append(int(str_gt_list[video_index][match_index_subject[video_index][sample_index][0]][-1])) # Get emotion of the match index
                    sample_count += 1
            UF1, UAR, F1_score = recognition_evaluation(str_gt_subject, str_pred_subject, label_dict, show=False)

        print('Cumulative result until subject %s:' % (subject_count+1))
        print('-----------------  Spotting   -----------------')
        print('Micro result: TP:%d FP:%d FN:%d AP[.5:.95]:%5.4f F1_score:%5.4f' % (TP_micro_total, FP_micro_total, FN_micro_total, AP_micro_total, F1_score_micro_total))
        print('Macro result: TP:%d FP:%d FN:%d AP[.5:.95]:%5.4f F1_score:%5.4f' % (TP_macro_total, FP_macro_total, FN_macro_total, AP_macro_total, F1_score_macro_total))
        print('Overall result: TP:%d FP:%d FN:%d AP[.5:.95]:%5.4f F1_score:%5.4f' % (TP_overall_total, FP_overall_total, FN_overall_total, AP_overall_total, F1_score_overall_total))
        print('-----------  Spot Then Recognize   ------------')
        print('Predicted    :', str_pred_subject)
        print('Ground Truth :', str_gt_subject)
        str_gt_all.extend(str_gt_subject)
        str_pred_all.extend(str_pred_subject)
        if subject_count + 1 == len(final_subjects):
            UF1, UAR, F1_score = recognition_evaluation(str_gt_all, str_pred_all, label_dict, show=True)
        else:
            UF1, UAR, F1_score = recognition_evaluation(str_gt_all, str_pred_all, label_dict, show=False)
        print('UF1:%5.4f, UAR:%5.4f, F1-score:%5.4f, STRS:%5.4f' % (UF1, UAR, F1_score, (F1_score_micro_total * F1_score)))
        print('----------------- Recognition -----------------')
        print('Predicted    :', pred)
        print('Ground Truth :', gt)
        cur_gt.extend(gt)
        cur_pred.extend(pred)
        if subject_count + 1 == len(final_subjects):
            UF1, UAR, F1_score = recognition_evaluation(cur_gt, cur_pred, label_dict, show=True)
        else:
            UF1, UAR, F1_score = recognition_evaluation(cur_gt, cur_pred, label_dict, show=False)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4), '| F1-Score:', round(F1_score, 4))
        print('\n')

        print('Done Subject', subject_count+1, ', spNO.', subjects_unique[subject_count])
    #     break

    end = time.time()
    print('Total time taken for training & testing: ' + str(end-start) + 's')

    return result_final, result1_final, cur_gt, cur_pred