# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import timm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization,feature_visualization_all
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 180  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m_ir = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m_rgb = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m_fus = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        """
        Args:
            x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)

        Return：
            if train:
                x (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            else:
                inference (tensor): (b, n_all_anchors, self.no)
                x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)
        """
        z = []  # inference output
        logits_ = []
        # output = x[3:]
        # x = x[:3]
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                logits = x[i][..., 5:]
                y = x[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)
                # logits_.append(logits.view(bs, -1, self.no - 5))
        return x if self.training else (torch.cat(z, 1), x)
        # return x if self.training else (torch.cat(z, 1),torch.cat(logits_, 1), x)
        # return (x,output) if self.training else (torch.cat(z, 1), x )
        # return (x, output) if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)

    # def forward(self, x):
    #     """
    #     Args:
    #         x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i), 共9个输入
    #
    #     Return：
    #         if train:
    #             x (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na 表示anchor scales的数量
    #         else:
    #             inference (tensor): (b, n_all_anchors, self.no)
    #             x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)
    #     """
    #     z_ir = []  # inference output
    #     z_rgb = []  # inference output
    #     z_fus = []  # inference output
    #     ir = x[:3]
    #     rgb = x[3:6]
    #     fus = x[6:9]
    #     logits_ = []
    #     for i in range(self.nl):
    #         ir[i] = self.m_ir[i](ir[i])  # conv
    #         bs, _, ny, nx = ir[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
    #         ir[i] = ir[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    #
    #         if not self.training:  # inference
    #             if self.onnx_dynamic or self.grid[i].shape[2:4] != ir[i].shape[2:4]:
    #                 self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
    #             logits = ir[i][..., 5:]
    #             y_ir = ir[i].sigmoid()  # (tensor): (b, self.na, h, w, self.no)
    #             if self.inplace:
    #                 y_ir[..., 0:2] = (y_ir[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 y_ir[..., 2:4] = (y_ir[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
    #                 xy = (y_ir[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 wh = (y_ir[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #                 y_ir = torch.cat((xy, wh, y_ir[..., 4:]), -1)
    #             z_ir.append(y_ir.view(bs, -1, self.no))  # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)
    #         rgb[i] = self.m_rgb[i](rgb[i])  # conv
    #         bs, _, ny, nx = rgb[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
    #         rgb[i] = rgb[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    #
    #         if not self.training:  # inference
    #             if self.onnx_dynamic or self.grid[i].shape[2:4] != rgb[i].shape[2:4]:
    #                 self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
    #             logits = rgb[i][..., 5:]
    #             y_rgb = rgb[i].sigmoid()  # (tensor): (b, self.na, h, w, self.no)
    #             if self.inplace:
    #                 y_rgb[..., 0:2] = (y_rgb[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 y_rgb[..., 2:4] = (y_rgb[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
    #                 xy = (y_rgb[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 wh = (y_rgb[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #                 y_rgb = torch.cat((xy, wh, y_rgb[..., 4:]), -1)
    #             z_rgb.append(y_rgb.view(bs, -1, self.no))  # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)
    #         fus[i] = self.m_fus[i](fus[i])  # conv
    #         bs, _, ny, nx = fus[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
    #         fus[i] = fus[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    #
    #         if not self.training:  # inference
    #             if self.onnx_dynamic or self.grid[i].shape[2:4] != fus[i].shape[2:4]:
    #                 self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
    #             logits = fus[i][..., 5:]
    #             y_fus = fus[i].sigmoid()  # (tensor): (b, self.na, h, w, self.no)
    #             if self.inplace:
    #                 y_fus[..., 0:2] = (y_fus[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 y_fus[..., 2:4] = (y_fus[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
    #                 xy = (y_fus[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 wh = (y_fus[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #                 y_fus = torch.cat((xy, wh, y_fus[..., 4:]), -1)
    #             z_fus.append(y_fus.view(bs, -1, self.no))  # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)
    #     return (ir,rgb,fus) if self.training else (torch.cat(z_ir, 1),torch.cat(z_rgb, 1),torch.cat(z_fus, 1),ir,rgb,fus)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
class Detect_rgb(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 180  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m_ir = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m_rgb = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m_fus = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        """
        Args:
            x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)

        Return：
            if train:
                x (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            else:
                inference (tensor): (b, n_all_anchors, self.no)
                x (list[P3_in,...]): torch.Size(b, c_i, h_i, w_i)
        """
        z = []  # inference output
        logits_ = []
        # output = x[3:]
        # x = x[:3]
        for i in range(self.nl):
            x[i] = self.m_rgb[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x[i](bs,self.no * self.na,20,20) to x[i](bs,self.na,20,20,self.no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                logits = x[i][..., 5:]
                y = x[i].sigmoid() # (tensor): (b, self.na, h, w, self.no)
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no)) # z (list[P3_pred]): Torch.Size(b, n_anchors, self.no)
                logits_.append(logits.view(bs, -1, self.no - 5))
        return x if self.training else (torch.cat(z, 1), x)
        return x if self.training else (torch.cat(z, 1),torch.cat(logits_, 1), x)
        # return (x,output) if self.training else (torch.cat(z, 1), x )
        # return (x, output) if self.training else (torch.cat(z, 1), torch.cat(logits_, 1), x)
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-18]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # m.inplace = self.inplacp
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            m.anchors /= m.stride.view(-1, 1, 1) # featuremap pixel
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
        m_rgb = self.model[-3]  # Detect()
        if isinstance(m_rgb, Detect_rgb):
            s = 256  # 2x min stride
            # m.inplace = self.inplacp
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m_rgb.stride = torch.Tensor([8.0, 16.0, 32.0])
            m_rgb.anchors /= m_rgb.stride.view(-1, 1, 1)  # featuremap pixel
            check_anchor_order(m_rgb)
            self.stride = m_rgb.stride
            self._initialize_biases_rgb()  # only run once
        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x,x2, augment=False, profile=False, visualize=False):
        """
        Args:
            x (tensor): (b, 3, height, width), RGB

        Return：
            if not augment:
                x (list[P3_out, ...]): tensor.Size(b, self.na, h_i, w_i, c), self.na means the number of anchors scales
            else:

        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self.forward_once(x, x2 , profile)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
    def forward_once(self, x, x2, profile=False,visualize = False):
        """
        :param x:          RGB Inputs
        :param x2:         IR  Inputs
        :param profile:
        :return:
        """
        y, dt = [], []  # outputs
        i = 0
        for m in self.model:
            # print("Moudle", i)
            if m.f != -1:  # if not from previous layer
                if m.f != -4:
                    # print(m)
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if m.f == -4:
                x = m(x2)
            else:
                # print(m.i)
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # print(len(y))
            # visualize = True
            if visualize:
                if visualize and (m.i == 4 or m.i == 14 or m.i == 20):
                    # feature_visualization(x, m.type, m.i, save_dir=Path('./keshihua/add'))
                    feature_visualization_all(x, m.type, m.i, save_dir=Path('./keshihua/aligned'))
            i += 1
        if profile:
            LOGGER.info('%.1fms total' % sum(dt))

        return x
    # def forward_once(self, x,x2, profile=False, visualize=False):
    #     y, dt = [], []  # outputs
    #     for m in self.model:
    #         if m.f != -1:  # if not from previous layer
    #             if m.f != -4:
    #                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
    #         if profile:
    #             self._profile_one_layer(m, x, dt)
    #         if hasattr(m, 'backbone') and m.f != -4:
    #             x = m(x)
    #             for _ in range(5 - len(x)):
    #                 x.insert(0, None)
    #             for i_idx, i in enumerate(x):
    #                 if i_idx in self.save:
    #                     y.append(i)
    #                 else:
    #                     y.append(None)
    #             x = x[-1]
    #         elif hasattr(m, 'backbone') and m.f == -4:
    #             x = m(x2)
    #             for _ in range(5 - len(x)):
    #                 x.insert(0, None)
    #             for i_idx, i in enumerate(x):
    #                 if i_idx in self.save:
    #                     y.append(i)
    #                 else:
    #                     y.append(None)
    #             x = x[-1]
    #         # elif m.f == -4:
    #         #     x = m(x2)
    #         #     y.append(x if m.i in self.save else None)  # save output
    #         else:
    #             x = m(x)  # run
    #             y.append(x if m.i in self.save else None)  # save output
    #         if visualize:
    #             feature_visualization(x, m.type, m.i, save_dir=visualize)
    #     return x
    # def forward_once(self, x1, x2, profile=False,visualize = False):
    #     """
    #     :param x:          RGB Inputs
    #     :param x2:         IR  Inputs
    #     :param profile:
    #     :return:
    #     """
    #     y, dt = [], []  # outputs
    #     i = 0
    #     for m in self.model:
    #         # print("Moudle", i)
    #         if m.f != -1:  # if not from previous layer
    #             if m.f != -4:
    #                 if m.f != -3:
    #                     if m.f != -2:
    #                         x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
    #         if profile:
    #             self._profile_one_layer(m, x, dt)
    #         if m.f == -2:
    #             # 将 x 和 x2 合并成一个张量 x3
    #             x3 = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0)), dim=0)
    #             # 取出合并后的张量中的各个部分
    #             x = m(x3)
    #         elif m.f == -4:
    #             x = m(x2)
    #         elif m.f == -3:
    #             x = m(x1)
    #         else:
    #             x = m(x)  # run
    #         y.append(x if m.i in self.save else None)  # save output
    #         # print(len(y))
    #         # visualize = True
    #         if visualize and (m.i == 5 or m.i == 15 or m.i == 21):
    #             feature_visualization(x, m.type, m.i, save_dir=Path(''))
    #         i += 1
    #     if profile:
    #         LOGGER.info('%.1fms total' % sum(dt))
    #
    #
    #     return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-3].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-18]  # Detect() module
        # for mi_ir, s in zip(m.m_ir, m.stride):  # from
        #     b = mi_ir.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        #     b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        #     b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
        #     mi_ir.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # for mi_rgb, s in zip(m.m_rgb, m.stride):  # from
        #     b = mi_rgb.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        #     b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        #     b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
        #     mi_rgb.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # for mi_fus, s in zip(m.m_fus, m.stride):  # from
        #     b = mi_fus.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        #     b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        #     b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
        #     mi_fus.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def _initialize_biases_rgb(self, cf=None):
        m_rgb = self.model[-3]  # Detect() module
        for mi, s in zip(m_rgb.m_rgb, m_rgb.stride):  # from
            b = mi.bias.view(m_rgb.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m_rgb.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def _print_biases(self):
        m = self.model[-18]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
        m_rgb = self.model[-3]  # Detect() module
        for mi in m_rgb.m_rgb:  # from
            b = mi.bias.detach().view(m_rgb.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
        # for mi_rgb in m.m_rgb:  # from
        #     b = mi_rgb.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
        #     LOGGER.info(
        #         ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi_rgb.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
        # for mi_ir in m.m_ir:  # from
        #     b = mi_ir.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
        #     LOGGER.info(
        #         ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi_ir.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
        # for mi_fus in m.m_fus:  # from
        #     b = mi_fus.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
        #     LOGGER.info(
        #         ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi_fus.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-18]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        m_rgb = self.model[-3]  # Detect()
        if isinstance(m_rgb, Detect):
            m_rgb.stride = fn(m_rgb.stride)
            m_rgb.grid = list(map(fn, m_rgb.grid))
            if isinstance(m_rgb.anchor_grid, list):
                m_rgb.anchor_grid = list(map(fn, m_rgb.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    no = na * (nc + 185)  # number of outputs = anchors * (classes + 185)
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost,Conv1,FeatureAlign,SCINet,FusionNet1w,Conv2,Conv3,STN_Net,C3_DCNV3]:
            if m is Conv1:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is Conv2:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is Conv3:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is FeatureAlign:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is SCINet:
                c1, c2 = 3, args[0]
                # print("focus c2", c2)
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is FusionNet1w:
                c1, c2 = 3, args[0]
                # print("focus c2", c2)
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            # elif m is fusionGAN:
            #     c1, c2 = 3, args[0]
            #     # print("focus c2", c2)
            #     if c2 != no:  # if not output
            #         c2 = make_divisible(c2 * gw, 8)
            #     args = [c1, c2, *args[1:]]
            elif m is STN_Net:
                c1, c2 = 3, args[0]
                # print("focus c2", c2)
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Detect_rgb:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Add:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Output:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Output1:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add_CrossPath:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            c2 = ch[f[0]]
            args = [c2, args[1]]
        elif m is CMF:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is DCNV3:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is DCnv2:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Guidefusion:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Frefusion:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add_fusion:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Only_ir:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is cross_fusion:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is FeatureAlign_V2:
            c2 = ch[f[0]]
            args = [c2]
        elif m is RgbIrSTN:
            if f == -4:
                c2 = 3
            else:
                c2 = ch[f[0]]
            args = [c2 * 2]
        elif m is ACF:
            c2 = sum([ch[f[0]], ch[f[1]]])
            args = [ch[f[0]], *args]
        elif m is LatentG:
            c2 = 512
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1 and x != -2)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
# def parse_model(d, ch):  # model_dict, input_channels(3)
#     # Parse a YOLOv5 model.yaml dictionary
#     LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
#     anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
#     if act:
#         Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
#         LOGGER.info(f"{colorstr('activation:')} {act}")  # print
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
#     aa = 0
#     is_backbone = False
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
#         try:
#             t = m
#             m = eval(m) if isinstance(m, str) else m  # eval strings
#         except:
#             pass
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except:
#                 args[j] = a
#
#         n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in {
#                 Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
#                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d,Conv1}:
#             if m is Conv1:
#                 c1, c2 = 3, args[0]
#                 if c2 != no:  # if not output
#                     c2 = make_divisible(c2 * gw, 8)
#                 args = [c1, c2, *args[1:]]
#             else:
#                 c1, c2 = ch[f], args[0]
#                 if c2 != no:  # if not output
#                     c2 = make_divisible(c2 * gw, 8)
#                 args = [c1, c2, *args[1:]]
#             if m in {BottleneckCSP, C3, C3TR, C3Ghost}:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum(ch[x] for x in f)
#         # TODO: channel, gw, gd
#         elif m is Detect:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is Add:
#             # print("ch[f]", f, ch[f[0]])
#             c2 = ch[f[0]]
#             args = [c2]
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         elif isinstance(m, str):
#             t = m
#             m = timm.create_model(m, pretrained=args[0], features_only=True)
#             c2 = m.feature_info.channels()
#         # elif m in {}:
#         #     m = m(*args)
#         #     c2 = m.channel
#         else:
#             c2 = ch[f]
#         if isinstance(c2, list):
#             is_backbone = True
#             m_ = m
#             m_.backbone = True
#             aa = aa + 4
#         else:
#             m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
#             t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum(x.numel() for x in m_.parameters())  # number params
#         m_.i, m_.f, m_.type, m_.np = i + aa if is_backbone else i, f, t, np  # attach index, 'from' index, type, number params
#         # print(m_.i)
#         LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
#         save.extend(x % (i + aa if is_backbone else i) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         if isinstance(c2, list):
#             ch.extend(c2)
#             for _ in range(5 - len(ch)):
#                 ch.insert(0, 0)
#         else:
#             ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/server08/ljc/yolov5_obb-master/models/multimode/ljc.yaml', help='model.yaml')
    parser.add_argument('--device', default='7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
