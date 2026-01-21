# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source1 =ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        source2 = ROOT / 'data/images' ,
        imgsz=(406, 519),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=True,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source1 = str(source1)
    save_img = not nosave and not source1.endswith('.txt')  # save inference images
    is_file = Path(source1).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source1.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source1.isnumeric() or source1.endswith('.txt') or (is_url and not is_file)
    source2 = str(source2)
    if is_url and is_file:
        source1 = check_file(source1)  # download
        source2 = check_file(source2)

    # Directories
    save_dir = (Path(project))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    # names = ['car', 'truck', 'bus', 'van', 'feright_car']
    # imgsz = (406, 519)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset_rgb = LoadStreams(source1, img_size=imgsz, stride=stride, auto=pt)
        dataset_ir = LoadStreams(source2, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset_rgb)  # batch_size
    else:
        dataset_rgb = LoadImages(source1, img_size=imgsz, stride=stride, auto=pt)
        dataset_ir = LoadImages(source2, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    # for path, im, im0s, vid_cap, s in dataset:
    for (path, im, im0s, vid_cap,s), (path_, im2, im0s_, vid_cap_,s_) in zip(dataset_rgb, dataset_ir):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im2 = torch.from_numpy(im2).to(device)
        im2 = im2.half() if half else im2.float()  # uint8 to fp16/32
        im2 /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im2.shape) == 3:
            im2 = im2[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im2,im, augment=augment, visualize=visualize)

        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        # pred: list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset_rgb.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset_rgb, 'frame', 0)
                p, im0_, frame = path, im0s_.copy(), getattr(dataset_ir, 'frame', 0)


            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset_rgb.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            annotator_ir = Annotator(im0_, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale polys from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)
                det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                class_counts = {name: 0 for name in names}
                # Write results
                log_file_path = str(save_dir / 'result_log.txt')  # 日志保存路径
                for *poly, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        poly = [int(p.item()) for p in poly]  # 或使用 round(float(p), 2) 保留小数
                        line = (cls, *poly, conf) if save_conf else (cls, *poly)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    cls_name = names[int(cls)]
                    cls_id = int(cls)
                    # 构建一行记录：x1 y1 x2 y2 x3 y3 x4 y4 类别名 类别ID 置信度
                    line = ' '.join(map(str, poly)) + f' {cls_name} {cls_id} {conf:.4f}'
                    # 打印
                    print(line)
                    # 写入 result_log.txt（追加）
                    with open(log_file_path, 'a') as f:
                        f.write(line + '\n')
                    # 增加对应类别的计数
                    class_counts[cls_name] += 1
                    if save_img or save_crop or view_img:  # Add poly to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator.poly_label(poly, label, color=colors(c, True))
                        annotator_ir.poly_label(poly, label, color=colors(c, True))
                        if save_crop: # Yolov5-obb doesn't support it yet
                            # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            pass

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            # 指定结果文件的路径
            result_file_path = str(save_dir / 'number.txt')
            # 打开文件，写入类别计数结果
            with open(result_file_path, 'w') as result_file:  # 'w'模式表示写入并覆盖已存在的内容
                for name, count in class_counts.items():
                    result_file.write(f"{name}: {count}\n")
            # Stream results
            im0 = annotator.result()
            im0_ = annotator_ir.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset_rgb.mode == 'image':
                    # save_path_rgb = save_path.split('.')[0] + '_rgb.' + save_path.split('.')[1]
                    # save_path_ir = save_path.split('.')[0] + '_ir.' + save_path.split('.')[1]
                    save_path_rgb = str(save_dir  / 'rgb.jpg')
                    save_path_ir = str(save_dir / 'ir.jpg')
                    cv2.imwrite(save_path_rgb, im0)
                    cv2.imwrite(save_path_ir, im0_)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,  help='model path(s)')
    parser.add_argument('--source1', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source2', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt',default='True',action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize' ,default=True,action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project',  help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    opt.weights = '/server08/ljc/OBB/yolov5_obb-master/runs/train/dronevehicle/Ours/weights/best.pt'
    opt.source1 = '/server08/ljc/OBB/yolov5_obb-master/dataset/DroneVehicle/images/train/rgb1/00001.jpg'
    opt.source2 = '/server08/ljc/OBB/yolov5_obb-master/dataset/DroneVehicle/images/train/ir1/00001.jpg'
    opt.project = '/server08/ljc/OBB/yolov5_obb-master/runs/detect'
    main(opt)