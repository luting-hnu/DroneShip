# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""
import yaml
import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

from utils.rboxs_utils import poly2hbb, rbox2poly

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets_GAI import create_dataloader, create_dataloader_rgb_ir
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, scale_polys, xywh2xyxy, xyxy2xywh, non_max_suppression_obb)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


# def save_one_json(predn, jdict, path, class_map):
def save_one_json(pred_hbbn, pred_polyn, jdict, path, class_map):
    """
    Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236, "poly": [...]}
    Args:
        pred_hbbn (tensor): (n, [poly, conf, cls])
        pred_polyn (tensor): (n, [xyxy, conf, cls])
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(pred_hbbn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(pred_polyn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[-1]) + 1],  # COCO's category_id start from 1, not 0
                      'bbox': [round(x, 1) for x in b],
                      'score': round(p[-2], 5),
                      'poly': [round(x, 1) for x in p[:8]],
                      'file_name': path.stem})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.01,  # confidence threshold
        iou_thres=0.4,  # NMS IoU threshold
        task='test',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # Dataloader
    if not training:
        # model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        val_path_rgb = data['test_rgb']
        val_path_ir = data['test_ir']
        # dataloader = create_dataloader(data[task], imgsz, batch_size, stride, names, single_cls, pad=pad, rect=pt,
        #                                workers=workers, prefix=colorstr(f'{task}: '))[0]
        dataloader = \
        create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, stride, names, single_cls, pad=pad,
                                 rect=pt,
                                 prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'HBBmAP@.5', '  HBBmAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # loss = torch.zeros(3, device=device)
    loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets_rgb,targets_ir, paths, shapes) in enumerate(pbar):
        # targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) θ ∈ [-pi/2, pi/2)
        # shapes (tensor): (b, [(h_raw, w_raw), (hw_ratios, wh_paddings)])
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets_rgb = targets_rgb.to(device)
            targets_ir = targets_ir.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        img_rgb = im[:, :3, :, :]
        img_ir = im[:, 3:, :, :]
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred_ir,pred_rgb = model(img_ir, img_rgb) if training else model(img_ir, img_rgb, augment=augment,val=True)
        out_ir, train_out_ir = pred_ir
        out_rgb, train_out_rgb = pred_rgb
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            # loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, theta
            loss += compute_loss([x.float() for x in train_out_rgb], targets_rgb)[1]  # box, obj, cls, theta
            # loss += compute_loss([x.float() for x in train_out_ir], targets_ir)[1]  # box, obj, cls, theta
            loss += compute_loss([x.float() for x in train_out_ir], targets_ir)[1]  # box, obj, cls, theta
            # loss += compute_loss([x.float() for x in train_out], targets_ir)[1]  # box, obj, cls, theta
        # NMS
        # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb_rgb = [targets_rgb[targets_rgb[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        lb_ir = [targets_ir[targets_ir[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        out_rgb = non_max_suppression_obb(out_rgb, conf_thres, iou_thres, labels=lb_rgb, multi_label=True,agnostic=single_cls)  # list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        out_ir = non_max_suppression_obb(out_ir, conf_thres, iou_thres, labels=lb_ir, multi_label=True,agnostic=single_cls)  # list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        dt[2] += time_sync() - t3

        # Metrics
        for si, (pred_rgb,pred_ir) in enumerate(zip(out_rgb,out_ir)):  # pred (tensor): (n, [xylsθ, conf, cls])
            labels_rgb = targets_rgb[targets_rgb[:, 0] == si, 1:7]  # labels (tensor):(n_gt, [clsid cx cy l s theta]) θ[-pi/2, pi/2)
            labels_ir = targets_ir[targets_ir[:, 0] == si, 1:7]  # labels (tensor):(n_gt, [clsid cx cy l s theta]) θ[-pi/2, pi/2)
            nl_rgb = len(labels_rgb)
            nl_ir = len(labels_ir)
            tcls = labels_ir[:, 0].tolist() if nl_ir else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]  # shape (tensor): (h_raw, w_raw)
            seen += 1

            if len(pred_ir) == 0:
                if nl_rgb:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                # pred[:, 5] = 0
                pred_rgb[:, 6] = 0
            poly_rgb = rbox2poly(pred_rgb[:, :5])  # (n, 8)
            pred_poly_rgb = torch.cat((poly_rgb, pred_rgb[:, -2:]), dim=1)  # (n, [poly, conf, cls])
            hbbox_rgb = xywh2xyxy(poly2hbb(pred_poly_rgb[:, :8]))  # (n, [x1 y1 x2 y2])
            pred_hbb_rgb = torch.cat((hbbox_rgb, pred_poly_rgb[:, -2:]), dim=1)  # (n, [xyxy, conf, cls])

            pred_polyn_rgb = pred_poly_rgb.clone()  # predn (tensor): (n, [poly, conf, cls])
            scale_polys(im[si].shape[1:], pred_polyn_rgb[:, :8], shape, shapes[si][1])  # native-space pred
            hbboxn_rgb = xywh2xyxy(poly2hbb(pred_polyn_rgb[:, :8]))  # (n, [x1 y1 x2 y2])
            pred_hbbn_rgb = torch.cat((hbboxn_rgb, pred_polyn_rgb[:, -2:]), dim=1)  # (n, [xyxy, conf, cls]) native-space pred
            if single_cls:
                # pred[:, 5] = 0
                pred_ir[:, 6] = 0
            poly_ir = rbox2poly(pred_ir[:, :5])  # (n, 8)
            pred_poly_ir = torch.cat((poly_ir, pred_ir[:, -2:]), dim=1)  # (n, [poly, conf, cls])
            hbbox_ir = xywh2xyxy(poly2hbb(pred_poly_ir[:, :8]))  # (n, [x1 y1 x2 y2])
            pred_hbb_ir = torch.cat((hbbox_ir, pred_poly_ir[:, -2:]), dim=1)  # (n, [xyxy, conf, cls])

            pred_polyn_ir = pred_poly_ir.clone()  # predn (tensor): (n, [poly, conf, cls])
            scale_polys(im[si].shape[1:], pred_polyn_ir[:, :8], shape, shapes[si][1])  # native-space pred
            hbboxn_ir = xywh2xyxy(poly2hbb(pred_polyn_ir[:, :8]))  # (n, [x1 y1 x2 y2])
            pred_hbbn_ir = torch.cat((hbboxn_ir, pred_polyn_ir[:, -2:]), dim=1)  # (n, [xyxy, conf, cls]) native-space pred

            # Evaluate
            if nl_rgb:
                # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                tpoly_rgb = rbox2poly(labels_rgb[:, 1:6])  # target poly
                tbox_rgb = xywh2xyxy(poly2hbb(tpoly_rgb))  # target  hbb boxes [xyxy]
                scale_coords(im[si].shape[1:], tbox_rgb, shape, shapes[si][1])  # native-space labels
                labels_hbbn_rgb = torch.cat((labels_rgb[:, 0:1], tbox_rgb), 1)  # native-space labels (n, [cls xyxy])
                correct_rgb = process_batch(pred_hbbn_rgb, labels_hbbn_rgb, iouv)
                if plots:
                    confusion_matrix.process_batch(pred_hbbn_rgb, labels_hbbn_rgb)
            else:
                correct_rgb = torch.zeros(pred_rgb.shape[0], niou, dtype=torch.bool)
            if nl_ir:
                # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                tpoly_ir = rbox2poly(labels_ir[:, 1:6])  # target poly
                tbox_ir = xywh2xyxy(poly2hbb(tpoly_ir))  # target  hbb boxes [xyxy]
                scale_coords(im[si].shape[1:], tbox_ir, shape, shapes[si][1])  # native-space labels
                labels_hbbn_ir = torch.cat((labels_ir[:, 0:1], tbox_ir), 1)  # native-space labels (n, [cls xyxy])
                correct_ir = process_batch(pred_hbbn_ir, labels_hbbn_ir, iouv)
                if plots:
                    confusion_matrix.process_batch(pred_hbbn_ir, labels_hbbn_ir)
            else:
                correct_ir = torch.zeros(pred_ir.shape[0], niou, dtype=torch.bool)
            # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            stats.append((correct_ir.cpu(), pred_poly_ir[:, 8].cpu(), pred_poly_ir[:, 9].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:  # just save hbb pred results!
                save_one_txt(pred_hbbn_ir, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                # LOGGER.info('The horizontal prediction results has been saved in txt, which format is [cls cx cy w h /conf/]')
            if save_json:  # save hbb pred results and poly pred results.
                save_one_json(pred_hbbn_ir, pred_polyn_ir, jdict, path, class_map)  # append to COCO-JSON dictionary
                # LOGGER.info('The hbb and obb results has been saved in json file')
            callbacks.run('on_val_image_end', pred_hbb_ir, pred_hbbn_ir, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels_rgb.jpg'  # labels
            Thread(target=plot_images, args=(img_rgb, targets_rgb, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred_rgb.jpg'  # predictions
            Thread(target=plot_images, args=(img_rgb, output_to_target(out_rgb), paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_labels_ir.jpg'  # labels
            Thread(target=plot_images, args=(img_ir, targets_ir, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred_ir.jpg'  # predictions
            Thread(target=plot_images, args=(img_ir, output_to_target(out_ir), paths, f, names), daemon=True).start()

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_obb_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
            LOGGER.info(
                '---------------------The hbb and obb results has been saved in json file-----------------------')

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/data_ship.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / '/server08/ljc/yolov5_obb-master/runs/train/exp1127/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', default=True, action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        # if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        if opt.conf_thres > 0.01:
            LOGGER.info(
                f'WARNING: In oriented detection, confidence threshold {opt.conf_thres} >> 0.01 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)