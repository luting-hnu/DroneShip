# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms_rotated
from mmdet.registry import DATASETS
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.utils import ProgressBar

from mmrotate.structures.bbox import rbbox_overlaps, qbox2rbox
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.1,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.2,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=0.1,
        help='nms IoU threshold, only applied when users want to change the'
             'nms IoU threshold.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0,
                               nms_iou_thr=None,
                               tp_iou_thr=0.1):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(dataset.metainfo['classes'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    std = 0
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['vis_instances']
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, num_classes, score_thr,
                             tp_iou_thr, nms_iou_thr)
        # std+=per_std
        prog_bar.update()
    # STD = std/len(results)
    # RMSE=torch.sqrt(STD[0].square()+STD[1].square())
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gts,
                         result,
                         num_classes,
                         score_thr=0.1,  # 0
                         tp_iou_thr=0.5,
                         nms_iou_thr=0.1

                         ):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    # true_positives = np.zeros(len(gts))
    gt_bboxes = []
    gt_labels = []
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    # gt_bboxes = np.array(gt_bboxes)
    # gt_labels = np.array(gt_labels)

    # unique_label = np.unique(result['labels'].numpy())
    # N=0
    # tmp=0

    # for det_label in unique_label:
    #     mask = (result['labels'] == det_label)

    #     det_boxes = result['bboxes'][mask].numpy()
    #     det_scores = result['scores'][mask].numpy()
    #     detection_classes= det_scores

    #     det_boxes= torch.Tensor( det_boxes )
    #     gt_bboxes= torch.Tensor( gt_bboxes )
    #     det_scores = torch.Tensor(det_scores )
    #     if gt_bboxes.shape[-1] == 8:
    #         gt_bboxes = qbox2rbox( gt_bboxes.int())
    # if nms_iou_thr:
    #     det_boxes, _ = nms_rotated(det_boxes, det_scores, nms_iou_thr)

    detections = torch.cat([result['bboxes'], result['scores'][:, None], result['labels'][:, None]], dim=1)
    gt_bboxes = torch.tensor(gt_bboxes)
    gt_labels = torch.tensor(gt_labels)

    if gt_bboxes.shape[-1] == 8:
        gt_bboxes = qbox2rbox(gt_bboxes.int())

    labels = torch.cat([gt_bboxes, gt_labels[:, None]], dim=1)
    # det_boxes, indice= nms_rotated(detections[:,:5], detections[:,5], nms_iou_thr)
    # label=detections[indice,-1]
    # detections =torch.cat([det_boxes,label[:,None]],dim=1)

    if detections is None:
        gt_classes = labels[:, -1].int()
        for gc in gt_classes:
            confusion_matrix[num_classes, gc] += 1  # background FN
        return
    detections = detections[detections[:, 5] > score_thr]
    gt_classes = labels[:, -1].int()
    detection_classes = detections[:, 6].int()
    iou = rbbox_overlaps(labels[:, :5].float(), detections[:, :5])
    x = torch.where(iou > tp_iou_thr)
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))

    n = matches.shape[0] > 0
    m0, m1, _ = matches.transpose().astype(int)
    for i, gc in enumerate(gt_classes):
        j = m0 == i
        if n and sum(j) == 1:
            confusion_matrix[detection_classes[m1[j]], gc] += 1  # correct
        else:
            confusion_matrix[num_classes, gc] += 1  # true background

    if n:
        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                confusion_matrix[dc, num_classes] += 1  # predicted background
    #     for i, det_bbox in enumerate(det_boxes):

    #         # score = det_bbox[5]
    #         score = det_scores[i]
    #         det_match = 0
    #         if score >= score_thr:
    #             for j, gt_label in enumerate(gt_labels):
    #                 if ious[i, j] >= tp_iou_thr:
    #                     det_match += 1
    #                     if gt_label == det_label:
    #                         true_positives[j] += 1  # TP
    #                         tmp += torch.square(det_bbox[:2] - gt_bboxes[j,:2])
    #                         N =N + 1
    #                     confusion_matrix[gt_label, det_label] += 1
    #             if det_match == 0:  # BG FP
    #                 confusion_matrix[-1, det_label] += 1
    # for num_tp, gt_label in zip(true_positives, gt_labels):
    #     if num_tp == 0:  # FN
    #         confusion_matrix[gt_label, -1] += 1
    # if N==0:
    #     return 0
    # else:
    #     std = torch.sqrt(tmp/N)
    #  return std


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='plasma'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """
    # normalize the confusion matrix
    # per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    # confusion_matrix = \
    #     confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}'.format(
                    int(confusion_matrix[
                            i,
                            j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()


def calculate_confusion_matrix_per_class(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    FP = np.zeros(num_classes)
    FN = np.zeros(num_classes)
    TP = np.zeros(num_classes)
    TN = np.zeros(num_classes)

    for i in range(num_classes):
        # for j in range(num_classes):
        # 计算假阳性和假阴性
        FP[i] += confusion_matrix[:, i].sum(axis=-1) - confusion_matrix[i, i]
        FN[i] += confusion_matrix[i, :].sum(axis=-1) - confusion_matrix[i, i]
        # 计算真阳性和真阴性
        TP[i] += confusion_matrix[i, i]
        # for j in range(num_classes):
        #     if i==j:
        #         continue
        #     TN[i]+=confusion_matrix[j, j]

        TN[i] += confusion_matrix[0:i, 0:i].sum() + confusion_matrix[0:i, i + 1:].sum() + confusion_matrix[i + 1:,
                                                                                          0:i].sum() + confusion_matrix[
                                                                                                       i + 1:,
                                                                                                       i + 1:].sum()  # 减去重复计算的部分
    return FP, FN, TP, TN


def main():
    register_all_modules()
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    results = load(args.prediction_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    confusion_matrix = calculate_confusion_matrix(dataset, results,
                                                  args.score_thr,
                                                  args.nms_iou_thr,
                                                  args.tp_iou_thr)
    ###############
    FP, FN, TP, TN = calculate_confusion_matrix_per_class(confusion_matrix)

    Precison = TP / (TP + FP)
    FA = FP / (FP + TN)

    Falsealarm = FA[:3].mean()
    P = Precison[:3].mean()
    print(' ')
    print('Each type of precision', Precison)
    print('Each type of falsealarm:', FA)
    print("Falsealarm:", Falsealarm)
    print("Precison:", P)

    # Precison = TP / (TP + FP)
    # FA = FP / (FP+TN)

    # print("False Positive:", FP)
    # print("False Negative:", FN)
    # print("True Positive:", TP)
    # print("True Negative:", TN)
    # print("Falsealarm:",FA)
    # print("Precison:",Precison)
    # print("RMSE:",RMSE)
    #############
    ######################
    # 2-TP/TN/FP/FN的计算
    # FP = confusion_matrix .sum(axis=0) - np.diag(confusion_matrix )
    # FN = confusion_matrix .sum(axis=1) - np.diag(confusion_matrix )
    # TP = np.diag(confusion_matrix )
    # TN = confusion_matrix .sum() - (FP + FN + TP)
    # FP = FP.astype(float)
    # FN = FN.astype(float)
    # TP = TP.astype(float)
    # TN = TN.astype(float)
    # print(TP)
    # print(TN)
    # print(FP)
    # print(FN)

    # # 3-其他的性能参数的计算
    # TPR = TP/(TP+FN) # Sensitivity/ hit rate/ recall/ true positive rate
    # TNR = TN/(TN+FP) # Specificity/ true negative rate
    # PPV = TP/(TP+FP) # Precision/ positive predictive value
    # NPV = TN/(TN+FN) # Negative predictive value
    # FPR = FP/(FP+TN) # Fall out/ false positive rate
    # FNR = FN/(TP+FN) # False negative rate
    # FDR = FP/(TP+FP) # False discovery rate
    # ACC = TP/(TP+FN) # accuracy of each class
    # print('TPR',TPR)
    # print('TNR',TNR)
    # print('PPV',PPV)
    # print('NPV',NPV)
    # print('FPR',FPR)
    # print('FNR',FNR)
    # print('FDR',FDR)
    # print('ACC',ACC)

    ######################

    plot_confusion_matrix(
        confusion_matrix,
        dataset.metainfo['classes'] + ('background',),
        save_dir=args.save_dir,
        color_theme=args.color_theme,
        show=args.show)


if __name__ == '__main__':
    main()
