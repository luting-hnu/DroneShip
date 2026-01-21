import os
import numpy as np
import torch
import argparse

# -----------------------------------------
# 1. batch_probiou 计算函数（基于论文：https://arxiv.org/pdf/2106.06072v1.pdf）

def _get_covariance_matrix(obb):
    """
    输入: obb [N, 5]，格式 (cx, cy, w, h, theta)，theta为弧度
    返回: a, b, c 计算cov矩阵用的参数，形状均为[N,1]
    """
    cx, cy, w, h, theta = obb[..., 0:1], obb[..., 1:2], obb[..., 2:3], obb[..., 3:4], obb[..., 4:5]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    a = (w / 2)**2 * cos_t**2 + (h / 2)**2 * sin_t**2
    b = (w / 2)**2 * sin_t**2 + (h / 2)**2 * cos_t**2
    c = ((w / 2)**2 - (h / 2)**2) * sin_t * cos_t

    return a, b, c


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    计算概率IoU，输入均为xywhr格式的obb，支持批量计算。

    obb1: (N,5) tensor或np.ndarray
    obb2: (M,5) tensor或np.ndarray

    返回：(N,M) tensor，IoU值
    """
    if isinstance(obb1, np.ndarray):
        obb1 = torch.from_numpy(obb1).float()
    if isinstance(obb2, np.ndarray):
        obb2 = torch.from_numpy(obb2).float()

    x1, y1 = obb1[..., 0:1], obb1[..., 1:2]
    x2, y2 = obb2[..., 0:1].T, obb2[..., 1:2].T  # 转置方便广播

    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    a1 = a1.unsqueeze(1)  # (N,1)
    b1 = b1.unsqueeze(1)
    c1 = c1.unsqueeze(1)
    x1 = x1.unsqueeze(1)
    y1 = y1.unsqueeze(1)

    a2 = a2.unsqueeze(0)  # (1,M)
    b2 = b2.unsqueeze(0)
    c2 = c2.unsqueeze(0)
    x2 = x2.unsqueeze(0)
    y2 = y2.unsqueeze(0)

    numerator = ((a1 + a2) * (y1 - y2)**2 + (b1 + b2) * (x1 - x2)**2)
    denominator = (a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps

    t1 = numerator / denominator * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denominator * 0.5

    t3_num = (a1 + a2) * (b1 + b2) - (c1 + c2)**2 + eps
    t3_den = 4 * torch.sqrt(torch.clamp(a1 * b1 - c1**2, min=0) * torch.clamp(a2 * b2 - c2**2, min=0)) + eps

    t3 = (t3_num / t3_den + eps).log() * 0.5

    bd = (t1 + t2 + t3).clamp(eps, 100)
    hd = torch.sqrt(1.0 - (-bd).exp() + eps)

    return 1 - hd  # (N,M)


# -----------------------------------------
# 2. 8点多边形转xywhr函数

def poly8_to_xywhr(poly8):
    """
    poly8: np.array shape (8,) => [x1,y1,x2,y2,x3,y3,x4,y4]
    返回: np.array shape (5,) => [cx, cy, w, h, theta(rad)]
    """
    pts = poly8.reshape(4, 2)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    edge1 = pts[1] - pts[0]
    edge2 = pts[2] - pts[1]
    w = np.linalg.norm(edge1)
    h = np.linalg.norm(edge2)
    theta = np.arctan2(edge1[1], edge1[0])
    return np.array([cx, cy, w, h, theta], dtype=np.float32)


# -----------------------------------------
# 3. 评价核心代码：替换原多边形IoU，改用概率IoU

def calcoverlaps_probIoU(BBGT_keep, bb):
    """
    计算预测bbox bb 与所有BBGT_keep的概率IoU。
    """
    BBGT_xywhr = np.array([poly8_to_xywhr(poly) for poly in BBGT_keep])
    bb_xywhr = poly8_to_xywhr(bb)
    obb1 = torch.tensor(BBGT_xywhr, dtype=torch.float32)  # (K,5)
    obb2 = torch.tensor(bb_xywhr, dtype=torch.float32).unsqueeze(0)  # (1,5)
    overlaps = batch_probiou(obb1, obb2).squeeze(1).cpu().numpy()
    return overlaps.tolist()


# -----------------------------------------
# 4. VOC eval代码（简化版）

def parse_gt(filename):
    """
    解析ground truth txt文件格式：
    每行格式：
    x1 y1 x2 y2 x3 y3 x4 y4 class_name [difficult]
    """
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return []

    objects = []
    with open(filename, 'r') as f:
        for line in f:
            splitlines = line.strip().split()
            if len(splitlines) < 9:
                continue
            obj_struct = {}
            obj_struct['name'] = splitlines[8]
            obj_struct['bbox'] = np.array([float(x) for x in splitlines[0:8]])
            obj_struct['difficult'] = int(splitlines[9]) if len(splitlines) == 10 else 0
            objects.append(obj_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            p = np.max(prec[rec >= t]) if np.any(rec >= t) else 0
            ap += p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    with open(imagesetfile, 'r') as f:
        imagenames = [x.strip() for x in f.readlines()]

    # 加载所有gt
    recs = {img: parse_gt(annopath.format(img)) for img in imagenames}

    class_recs = {}
    npos = 0
    for img in imagenames:
        objs = [obj for obj in recs[img] if obj['name'] == classname]
        bbox = np.array([obj['bbox'] for obj in objs])
        difficult = np.array([obj['difficult'] for obj in objs]).astype(bool)
        det = [False] * len(objs)
        npos += sum(~difficult)
        class_recs[img] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # 加载检测结果
    with open(detpath.format(classname), 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split() for x in lines]
    image_ids = [x[0].split('.')[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:10]] for x in splitlines])  # 8点坐标

    # 按置信度排序
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind]
    image_ids = [image_ids[i] for i in sorted_ind]
    confidence = confidence[sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d]

        ovmax = -np.inf
        BBGT = R['bbox']

        if BBGT.size > 0:
            # 先用包围盒简单过滤
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) * (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            if len(BBGT_keep) > 0:
                overlaps_poly = calcoverlaps_probIoU(BBGT_keep, bb)
                ovmax = np.max(overlaps_poly)
                jmax = np.argmax(overlaps_poly)
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = True
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


# -----------------------------------------
# 5. 主函数示例

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detpath', type=str, default='/server08/ljc/OBB/yolov5_obb-master/runs/val/dronevehicle/stage1/last_predictions_Txt/Task1_{:s}.txt', help='检测结果路径格式')
    # parser.add_argument('--detpath', type=str,default='/server08/ljc/OBB/C2Former-main/python_file/result/Task1_{:s}.txt',help='检测结果路径格式')
    parser.add_argument('--annopath', type=str, default='/server08/ljc/OBB/yolov5_obb-master/dataset/DroneVehicle/labelTxt/test/ir1/{:s}.txt', help='GT标注路径格式')
    parser.add_argument('--imagesetfile', type=str, default='/server08/ljc/OBB/yolov5_obb-master/dataset/DroneVehicle/imgnamefile.txt', help='图片列表文件')
    parser.add_argument('--classes', nargs='+', default=['car', 'truck', 'bus', 'van', 'feright_car'], help='类别列表')
    parser.add_argument('--ovthresh', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--use_07_metric', action='store_true', help='是否使用VOC2007 AP计算方法')
    args = parser.parse_args()

    map_total = 0
    for classname in args.classes:
        print(f"Evaluating class {classname} ...")
        rec, prec, ap = voc_eval(args.detpath, args.annopath, args.imagesetfile, classname, ovthresh=args.ovthresh,
                                use_07_metric=args.use_07_metric)
        print(f"AP for {classname}: {ap:.4f}")
        map_total += ap
    map_total /= len(args.classes)
    print(f"mAP = {map_total:.4f}")