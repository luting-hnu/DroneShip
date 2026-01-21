# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import os
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
import torch.nn.functional as F
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import fvcore.nn.weight_init as weight_init
from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync
from timm.models.layers import trunc_normal_
from torchvision.ops import DeformConv2d
from torch.cuda.amp import autocast, GradScaler
# from models.ops_dcnv3.modules import DCNv3
from models.SE import SEAttention
import matplotlib.pyplot as plt
from torch.nn import init, Sequential
# from DCNv2.dcn_v2 import DCN as dcn_v2
# from .irnn import irnn
# import kornia.filters as KF
# import kornia.utils as KU
ConvTranspose2d = torch.nn.ConvTranspose2d
BatchNorm2d = torch.nn.BatchNorm2d
interpolate = F.interpolate
Linear = torch.nn.Linear


def visualize_tensor(x, save_path):
    # Ensure the save directory exists
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    # Extract tensor shape
    batch, channels, height, width = x.shape
    # Proceed only if the feature map has spatial dimensions
    if height > 1 and width > 1:
        # Sum across channels to visualize as a single image
        img = x[0].cpu().transpose(0, 1).sum(1).detach().numpy()
        # Normalize the image to [0, 1] for better visualization
        img = (img - img.min()) / (img.max() - img.min())
        # Save the image
        plt.imsave(save_path, img, cmap='viridis')  # Use a colormap for better visualization
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat
# class Conv2d(torch.nn.Conv2d):
#     # A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
#     def __init__(self, *args, **kwargs):
#         norm = kwargs.pop("norm", None)
#         activation = kwargs.pop("activation", None)
#         super().__init__(*args, **kwargs)
#
#         self.norm = norm
#         self.activation = activation
#
#     def forward(self, x):
#         x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Convrelu_1(nn.Module):
    # convolution
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(Convrelu_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight.data)
    def forward(self, x):
        return F.relu(self.conv(x))
class Conv_3(nn.Module):
    # convolution
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(Conv_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight.data)
    def forward(self, x):
        return self.conv(x)
class Conv1(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Conv2(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        ir = x[0]
        return self.act(self.bn(self.conv(ir)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Conv3(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        vis = x[1]
        return self.act(self.bn(self.conv(vis)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
class Illumination(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Illumination, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
    def forward(self , x):
        batch_size = x.shape[0]
        vis_light_weight = torch.mean(x, dim=1)
        exp_input = -(10 * vis_light_weight + -3)
        exp_input = exp_input.to('cuda')
        y = 1 / (1 + torch.exp(exp_input))
        # w_vis = y.clone().detach().to(device='cuda', dtype=torch.float16).view(batch_size, 1, 1, 1)
        return y

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False):
        # Usage:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript
        #   CoreML:                 *.mlmodel
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        #   TensorRT:               *.engine
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix = Path(w).suffix.lower()
        suffixes = ['.pt', '.torchscript', '.onnx', '.engine', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, jit, onnx, engine, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local

        if jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '8.0.0', verbose=True)  # version requirement
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        else:  # TensorFlow model (TFLite, pb, saved_model)
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif saved_model:
                LOGGER.info(f'Loading {w} for TensorFlow saved_model inference...')
                import tensorflow as tf
                model = tf.keras.models.load_model(w)
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    import tflite_runtime.interpreter as tfli
                    delegate = {'Linux': 'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
                else:
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, img_rgb,img_ir, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = img_rgb.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(img_rgb,img_ir) if self.jit else self.model(img_rgb,img_ir, augment=augment, visualize=visualize)
            # return y if val else (y[0],y[1],y[2])
            return y if val else y[0]
            # return y if val else y[0][0],y[1][0]
        elif self.coreml:  # CoreML
            im = img_rgb.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
            conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
            y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
        elif self.onnx:  # ONNX
            img_rgb = img_rgb.cpu().numpy()  # torch to numpy
            img_ir = img_ir.cpu().numpy()
            if self.dnn:  # ONNX OpenCV DNN
                self.net.setInput(im)
                y = self.net.forward()
            else:  # ONNX Runtime
                y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img_rgb,self.session.get_inputs()[1].name:img_ir})[0]
        elif self.engine:  # TensorRT
            assert img_rgb.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(img_rgb.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        else:  # TensorFlow model (TFLite, pb, saved_model)
            im = img_rgb.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.pb:
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.saved_model:
                y = self.model(im, training=False).numpy()
            elif self.tflite:
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., 0] *= w  # x
            y[..., 1] *= h  # y
            y[..., 2] *= w  # w
            y[..., 3] *= h  # h
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.engine or self.onnx:  # warmup types
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
class ConvLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, norm=None, activation='LReLU', kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        model = []
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)]
        if norm == 'Batch':
            model += [nn.BatchNorm2d(out_channels)]
        if activation == 'LReLU': ## 默认使用LeakyReLU作为激活函数
            model += [nn.LeakyReLU(inplace=True)]
        elif activation == 'Sigmoid':
            model += [nn.Sigmoid()]
        elif activation == 'ReLU':
            model += [nn.ReLU()]
        elif activation == 'Tanh':
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        model = []
        out_channels = int(in_channels / 2)
        model += [ConvLeakyRelu2d(2 * in_channels, out_channels)]
        model += [ConvLeakyRelu2d(out_channels, out_channels)]
        model += [ConvLeakyRelu2d(out_channels, 4, activation='Sigmod', kernel_size=3, padding=1, stride=1)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        out = self.model(x)
        return out
class FusionNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FusionNet, self).__init__()
        channels = [8, 16, 16, 32]
        encoder_ir = []
        encoder_ir += [ConvLeakyRelu2d(in_channels, channels[0])]
        encoder_ir += [ConvLeakyRelu2d(channels[0], channels[1])]
        encoder_ir += [ConvLeakyRelu2d(channels[1], channels[2])]
        encoder_ir += [ConvLeakyRelu2d(channels[2], channels[3])]
        self.encoder_ir = nn.Sequential(*encoder_ir)
        encoder_vi = []
        encoder_vi += [ConvLeakyRelu2d(in_channels, channels[0])]
        encoder_vi += [ConvLeakyRelu2d(channels[0], channels[1])]
        encoder_vi += [ConvLeakyRelu2d(channels[1], channels[2])]
        encoder_vi += [ConvLeakyRelu2d(channels[2], channels[3])]
        self.encoder_vi = nn.Sequential(*encoder_vi)
        decoder = []
        decoder += [ConvLeakyRelu2d(channels[3], channels[2])]
        decoder += [ConvLeakyRelu2d(channels[2], channels[1])]
        decoder += [ConvLeakyRelu2d(channels[1], channels[0])]
        decoder += [ConvLeakyRelu2d(channels[0], out_channels,activation='Tanh')]
        self.decoder = nn.Sequential(*decoder)
        self.SAM = SAM(channels[3], channels[3], 1)
        # self.SAM_vi = SAM(channels[3], channels[3], 1)
    def forward(self, image_ir,image_vi, eps=1e-6):
        # split data into RGB and INF
        features_ir = self.encoder_ir(image_ir)
        features_vi = self.encoder_vi(image_vi)
        attention_ir = self.SAM(torch.cat([features_ir, features_vi], dim=1))
        # attention_vi = self.SAM_vi(features_vi)
        # features_fused = attention_ir * features_ir + (1- attention_ir) * features_vi
        # features_fused = features_ir * (attention_ir / (attention_vi + attention_ir)) + features_vi * (attention_vi / (attention_vi + attention_ir))
        features_fused = features_ir.mul(attention_ir) + features_vi.mul(1 - attention_ir)
        image_fused = self.decoder(features_fused)
        image_fused = (image_fused+1)/2
        return image_fused
class ResConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, dilation=1, norm=None,):
        super(ResConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        # elif == 'Group'
    def forward(self, x):
        return self.model(x)+x
class Feature_extractor_unshare(nn.Module):
    def __init__(self,depth,base_ic,base_oc,base_dilation,norm):
        super(Feature_extractor_unshare,self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i%2==1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(ResConv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            else:
                feature_extractor.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            if i%2==1 and i<depth-1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x
class ResConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, dilation=1, norm=None,):
        super(ResConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)+x
class DispRefiner(nn.Module):
    def __init__(self, channel, dilation=1, depth=4):
        super(DispRefiner, self).__init__()
        self.preprocessor = nn.Sequential(
            Conv2d(channel, channel, 3, dilation=dilation, padding=dilation, norm=None, act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel * 2, channel * 2, 3, padding=1),
                                            Conv2d(channel * 2, channel, 3, padding=1, norm=None, act=None))
        oc = channel
        ic = channel + 2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth - 1):
            oc = oc // 2
            estimator.append(
                Conv2d(ic, oc, kernel_size=3, stride=1, padding=dilation, dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc, 2, kernel_size=3, padding=1, dilation=1, act=None, norm=None))
        # estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)
    def forward(self, feat1, feat2, disp):
        b = feat1.shape[0]
        feat = torch.cat([feat1, feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b], feat[b:]], dim=1))
        corr = torch.cat([feat, disp], dim=1)
        delta_disp = self.estimator(corr)
        disp = disp + delta_disp
        return disp
def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out
def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr
class FeatureAlign(nn.Module):
    def __init__(self,c1,c2):
        super(FeatureAlign, self).__init__()
        self.DM = DenseMatcher()
        self.FN = FusionNet
        self.ST = SpatialTransformer(256, 256, True)
    def forward(self, x):
        vi = x[0]
        ir = x[1]
        disp = self.DM(ir, vi)['ir2vis']
        vi_reg = self.ST(vi, disp)
        batch_size = vi.shape[0]
        for i in range(batch_size):
            ir_reg = vi_reg[i:i + 1]  # 获取第i张图像
            vis_image = vi[i:i +1]
            detached_tensor = ir_reg.detach()
            ir_reg_np = detached_tensor[0].permute(1, 2, 0).cpu().numpy()  # 转换为 NumPy 数组
            vis_image_np = vis_image[0].permute(1, 2, 0).cpu().numpy()
            # Convert the NumPy array to a PIL Image
            ir_reg_image = Image.fromarray((ir_reg_np * 255).astype(np.uint8))
            vis_image = Image.fromarray((vis_image_np * 255).astype(np.uint8))
            # # Save the image with a unique name or index
            ir_reg_image.save(f'vi_reg.jpg')
            vis_image.save(f'vi.jpg')

        # vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)
        # fu = self.FN(ir_reg[:, 0:1], vi_Y)
        # fu = YCbCr2RGB(fu, vi_Cb, vi_Cr)
        return vi
class Conv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, dilation=1, norm=None, act=nn.LeakyReLU,bias=False):
        super(Conv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        if act is nn.LeakyReLU:
            model += [act(negative_slope=0.1,inplace=True)]
        elif act is None:
            model +=[]
        else:
            model +=[act()]
        self.model = nn.Sequential(*model)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)
class SCINet(nn.Module):
    def __init__(self, channels=3, layers=3):
        super(SCINet, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1)

        return illu

def RGB2YCbCr_batch(RGB_images):
    ## RGB_images shape: [batch_size, height, width, channels]
    test_num1 = 16.0 / 255.0
    test_num2 = 128.0 / 255.0
    R = RGB_images[:, :, :, 0:1]
    G = RGB_images[:, :, :, 1:2]
    B = RGB_images[:, :, :, 2:3]
    Y = 0.257 * R + 0.564 * G + 0.098 * B + test_num1
    Cb = - 0.148 * R - 0.291 * G + 0.439 * B + test_num2
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + test_num2
    return Y, Cb, Cr
def feature_visual(in_feature):
    # in_feature = torch.max(in_feature, dim=1)
    # in_feature = torch.squeeze(in_feature[0], dim=0)
    in_feature = torch.mean(in_feature, dim=1)
    in_feature = torch.squeeze(in_feature, dim=0)

    in_feature = in_feature.cpu().detach().numpy()
    in_feature = 255 * (in_feature - np.min(in_feature)) / (np.max(in_feature) - np.min(in_feature))
    in_feature = in_feature.astype(np.uint8)
    in_feature = cv2.applyColorMap(in_feature, cv2.COLORMAP_JET)
    return in_feature
class FusionNet1w(nn.Module):
    def __init__(self,c1,c2):
        super(FusionNet1w, self).__init__()
        self.FE_R = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.FE_T = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.de = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        ir = x[0]
        vis = x[1]
        ir_single = torch.mean(ir, dim=1, keepdim=True)
        Y, Cb, Cr = RGB2YCrCb(vis)
        RGB_1 = self.FE_R(Y)
        TIR_1 = self.FE_T(ir_single)
        fuse = RGB_1 + TIR_1
        att = self.de(fuse)
        w_vi = att[:, 0, ...].unsqueeze(1)
        w_ir = att[:, 1, ...].unsqueeze(1)
        # w_ir = 1 - w_vi
        RGB1 = torch.mul(Y, w_vi)
        TIR1 = torch.mul(ir_single, w_ir)
        # RGB2 = torch.mul(x, att_r2)
        # TIR2 = torch.mul(y, att_t2)
        att_r1_visula = w_vi
        att_t1_visula = w_ir
        att_r1_visula = feature_visual(att_r1_visula)
        att_t1_visula = feature_visual(att_t1_visula)
        cv2.imwrite(os.path.join('/home/server08/ljc/yolov5_obb-master/keshihua/w_ir.jpg' ), att_t1_visula)
        cv2.imwrite(os.path.join('/home/server08/ljc/yolov5_obb-master/keshihua/w_vi.jpg' ), att_r1_visula)
        out1 = RGB1 + TIR1
        # out2 = RGB2 + TIR2
        out = out1
        vis = torch.mul(vis , w_vi)
        ir = torch.mul(ir , w_ir)
        # out_color = YCbCr2RGB(out,Cb, Cr)
        # batch_size = out_color.shape[0]
        # for i in range(batch_size):
        #     img = out_color[i:i + 1]  # 获取第i张图像
        #     vis_image_np = img[0].permute(1, 2, 0).cpu().numpy()
        #     # Convert the NumPy array to a PIL Image
        #     vis_image = Image.fromarray((vis_image_np * 255).astype(np.uint8))
        #     # # Save the image with a unique name or inde
        #     vis_image.save(f'output.jpg')
        # output = [w_ir , w_vi , out]
        output = [ir, vis, out]
        return output
class FW(nn.Module):
    def __init__(self, int_c, out_c):
        super(FW, self).__init__()
        self.conv_R = nn.Sequential(
            nn.Conv2d(int_c, out_c//8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_T =  nn.Sequential(
            nn.Conv2d(int_c, out_c//8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.R = nn.Conv2d(1, out_c//8, kernel_size=1, stride=1, padding=0)
        self.T = nn.Conv2d(1, out_c//8, kernel_size=1, stride=1, padding=0)
        self.conv_recR = nn.Conv2d(out_c//8, out_c, kernel_size=1, stride=1, padding=0)
        self.conv_recT = nn.Conv2d(out_c // 8, out_c, kernel_size=1, stride=1, padding=0)
        self.gamma_rf = nn.Parameter(torch.zeros(1))
        self.gamma_tf = nn.Parameter(torch.zeros(1))

    def forward(self, x, y, R_w, T_w):
        size = x.size()[2:]
        RGB = self.conv_R(x)
        TIR = self.conv_T(y)
        R_w = self.R(R_w)
        T_w = self.T(T_w)
        R_w = F.interpolate(R_w, size=size, mode='bilinear', align_corners=True)
        T_w = F.interpolate(T_w, size=size, mode='bilinear', align_corners=True)
        rec_R = torch.mul(RGB, R_w)
        rec_T = torch.mul(TIR, T_w)
        rec_R = self.conv_recR(rec_R)
        rec_T = self.conv_recT(rec_T)
        rec_R = rec_R*self.gamma_rf + x
        rec_T = rec_T*self.gamma_tf + y
        return rec_R, rec_T
class Discriminator_i(nn.Module):
    def __init__(self):
        super(Discriminator_i, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid())
    def forward(self,i):
        batch_szie = i.shape[0]
        i = self.conv1(i)
        i = self.conv2(i)
        i = self.conv3(i)
        i = i.view((batch_szie,-1))
        i = self.fc(i)
        return i
class Discriminator_v(nn.Module):
    """docstring for Discriminator_v"""
    def __init__(self):
        super(Discriminator_v, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,16,3,2,1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv3=nn.Sequential(
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.fc=nn.Sequential(
            nn.Linear(65536,1),
            nn.Sigmoid())

    def forward(self,v):
        batch_szie=v.shape[0]
        v = self.conv1(v)
        v = self.conv2(v)
        v = self.conv3(v)
        v = v.view((batch_szie,-1))
        v = self.fc(v)
        return v
class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        output = torch.add(x[0], x[1])
        # output = x[0]
        return output
class I_Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(I_Add, self).__init__()
        self.arg = arg
        self.x0 = nn.Parameter(torch.tensor(-160.0))  # 偏移量
        self.k = nn.Parameter(torch.tensor(0.01))  # 斜率
        self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1)  # 输入通道数为1，输出通道数为1
        self.sigmoid = nn.Sigmoid()  # 用于将输出值限制在0到1之间
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ir = x[0]
        output_size = ir.size()[2:]

        vis = x[1]
        ill = x[2]
        # visualize_tensor(ill, save_path='./ljc/ill.jpg')
        ill = 1 / (1 + torch.exp(-self.k * (ill - self.x0)))
        # ill_rgb =1-ill_ir
        ill= F.adaptive_avg_pool2d(ill, output_size)  # 使用自适应平均池化调整大小
        ill = self.conv1x1(ill)
        ill = self.sigmoid(ill)
        w_ir =  ill
        w_rgb = 1 - ill
        # visualize_tensor(w_rgb, save_path='./ljc/rgb.jpg')
        # visualize_tensor(w_ir, save_path='./ljc/ir.jpg')
        # print(self.k)
        output = w_rgb * vis + w_ir * ir
        return output
def save_feature_map(tensor, save_path, normalize=True, cmap='viridis'):
    """
    将单通道特征图保存为图像文件。

    参数:
        tensor (torch.Tensor): 单通道特征图张量，形状为 (1, 1, H, W) 或 (H, W)。
        save_path (str or Path): 保存图像的路径。
        normalize (bool): 是否对特征图进行归一化处理（默认为 True）。
        cmap (str): 使用的颜色映射（默认为 'viridis'，其他可选如 'gray', 'jet', 'hot', 'coolwarm' 等）。
    """
    # 确保输入是单通道特征图
    if len(tensor.shape) == 4:  # (batch, channels, height, width)
        assert tensor.shape[0] == 1 and tensor.shape[1] == 1, "输入张量必须是单通道特征图"
        img = tensor.squeeze().cpu().detach().numpy()  # 去掉批量和通道维度
    elif len(tensor.shape) == 2:  # (height, width)
        img = tensor.cpu().detach().numpy()
    else:
        raise ValueError("输入张量的形状必须是 (1, 1, H, W) 或 (H, W)")

    # 归一化特征图（可选）
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())

    # 确保保存路径的目录存在
    save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 imshow 保存图像
    plt.figure()
    plt.imshow(img, cmap=cmap)  # 使用指定的颜色映射
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像窗口

    print(f"特征图已保存到：{save_path}")


class Add_fusion(nn.Module):
    #  Add two tensors
    def __init__(self, num_channels):
        super(Add_fusion, self).__init__()
        # self.R = nn.Conv2d(1, out_c // 8, kernel_size=1, stride=1, padding=0)
        # self.T = nn.Conv2d(1, out_c // 8, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        ir = x[0]
        vi = x[1]
        size = vi.size()[2:]
        output = x[2]
        w_ir = output[0]
        w_vi = output[1]
        w_vi = F.interpolate(w_vi, size=size, mode='bilinear', align_corners=True)
        w_ir = F.interpolate(w_ir, size=size, mode='bilinear', align_corners=True)
        # w_ir = self.conv(w_ir)
        # w_vi = self.conv(w_vi)
        rec_vi = torch.mul(vi, w_vi)
        rec_ir = torch.mul(ir, w_ir)
        output = torch.add(rec_vi, rec_ir)
        # output = ir
        # ir_feature = ir
        # vi_feature = vi
        # output_feature = output
        # att_r1_visula = w_vi
        # att_t1_visula = w_ir
        # ir_feature = 100 ** ir_feature
        # vi_feature = 100 ** vi_feature
        # att_r1_visula = feature_visual(att_r1_visula)
        # att_t1_visula = feature_visual(att_t1_visula)
        # ir_feature = feature_visual(ir_feature)
        # vi_feature = feature_visual(vi_feature)
        # output_feature = feature_visual(output_feature)
        # cv2.imwrite(os.path.join('/home/server08/ljc/yolov5_obb-master/keshihua/w_ir.jpg' ), att_t1_visula)
        # cv2.imwrite(os.path.join('/home/server08/ljc/yolov5_obb-master/keshihua/w_vi.jpg' ), att_r1_visula)
        # cv2.imwrite(os.path.join('/home/server08/ljc/yolov5_obb-master/keshihua/ir_feature.jpg'), ir_feature)
        # cv2.imwrite(os.path.join('/home/server08/ljc/yolov5_obb-master/keshihua/vi_feature.jpg'), vi_feature)
        # cv2.imwrite(os.path.join('/home/server08/ljc/yolov5_obb-master/keshihua/output_fusion.jpg'), output_feature)
        return output
class LJC(nn.Module):
    #  Add two tensors
    def __init__(self, in_channel):
        super(LJC, self).__init__()
        self.mam = MAM2(in_channel)
        self.fuse = FRM(in_channel)
    def forward(self, x):
        ir = x[0]
        rgb = x[1]
        x = [rgb , ir]
        grgb = self.mam(x)
        # final = self.fuse(grgb, ir)
        return torch.add(ir,grgb)
class Only_ir(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Only_ir, self).__init__()
        self.arg = arg

    def forward(self, x):
        return x[0]
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
class MAM2(nn.Module):
    def __init__(self, in_channel):
        super(MAM2, self).__init__()
        self.channel264 = nn.Sequential(
            Conv(in_channel, in_channel//2, 3, 2, 1),
            convblock(in_channel//2, in_channel//4, 3, 1, 1),
            convblock(in_channel//4, in_channel//8, 3, 1, 0),
            convblock(in_channel//8, in_channel//16, 3, 1, 1),
            convblock(in_channel//16, 16, 1, 1, 1),
        )
        self.xy = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 2, 1, 1, 0)
        )
        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, 1, 1, 0)
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, 1, 1, 0)
        )
        # Start with identity transformation
        self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.xy[-1].bias.data.zero_()
        self.scale1[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale1[-1].bias.data.zero_()
        self.scale2[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale2[-1].bias.data.zero_()
        # self.fus1 = Conv(in_channel * 2, in_channel, 1, 1, 0)
    def forward(self, x):
        gr = x[0]
        gt = x[1]
        n1 = self.channel264(gt)
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
        shift_xy = self.xy(n1)
        shift_s1 = self.scale1(n1)
        shift_s2 = self.scale2(n1)
        bsize = shift_xy.shape[0]
        identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1).cuda()
        identity_theta[:, :, 2] += shift_xy.squeeze()
        identity_theta[:, :1, :1] += shift_s1.squeeze(2)
        identity_theta[:, 1, 1] += shift_s2.squeeze()
        wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), gt.size(), align_corners=True).permute(0, 3, 1,2)
        wrap_grid = wrap_grid.to(gr.dtype)
        wrap_gr = F.grid_sample(gr, wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
        return wrap_gr
class FRM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FRM, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        # out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        x1 = x1 + self.lambda_c * channel_weights[0] * x1
        # out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        x2 = x2 + self.lambda_c * channel_weights[1] * x2
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_s * spatial_weights[0] * x1
        out_x2 = x2 + self.lambda_s * spatial_weights[1] * x2
        out = out_x1 + out_x2
        return out
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        # x = torch.cat((x1, x2), dim=1)
        # avg1 = self.avg_pool(x1).view(B, self.dim)
        avg1 = torch.mean(x1, dim=[2, 3], keepdim=True).view(B, self.dim)
        avg2 = torch.mean(x2, dim=[2, 3], keepdim=True).view(B, self.dim)
        # avg2 = self.avg_pool(x2).view(B, self.dim)
        max1 = self.max_pool(x1).view(B, self.dim)
        max2 = self.max_pool(x2).view(B, self.dim)
        avg = avg1+avg2
        max = max1+max2
        y = torch.cat((max, avg), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights
class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class STN_Net(nn.Module):
    def __init__(self, C1, C2):
        super(STN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Localisation net
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((10, 10))
        )
        # Create fc_loc based on xs size
        self.fc_loc = nn.Sequential(
            nn.Linear(10*10*10, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # Placeholder for fc_loc
    def stn(self, ir, vis):
        xs = self.localization(ir)
        # size_without_first_dim = xs.size()[1:].numel()
        xs = xs.view(-1, 10*10*10)

        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, ir.size(), align_corners=True)
        x = F.grid_sample(vis, grid, align_corners=True)

        return x

    def forward(self, x):
        ir = x[0]
        vis = x[1]
        print
        vis_change = self.stn(ir, vis)
        return vis_change


class DCnv2(nn.Module):
    def __init__(self, c2, k=3, s=1, p=1, g=1, act=True):
        super(DCnv2, self).__init__()
        # self.conv1 = nn.Conv2d(c1, c2, kernel_size=k, stride=1, padding=p, groups=g, bias=False)
        deformable_groups = 1
        offset_channels = 18
        mask_channels = 9
        self.conv2_offset = nn.Conv2d(c2, deformable_groups * offset_channels, kernel_size=k, stride=s, padding=p)
        self.conv2_mask = nn.Conv2d(c2, deformable_groups * mask_channels, kernel_size=k, stride=s, padding=p)
        # init_mask = torch.Tensor(np.zeros([mask_channels, 3, 3, 3])) + np.array([0.5])
        # self.conv2_mask.weight = torch.nn.Parameter(init_mask)
        self.conv2 = DeformConv2d(c2, c2, kernel_size=k, stride=s, padding=1, bias=True)
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.act1 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.cmf = CMF(c2)
        self.cbam = CBAM(c2)
        self.guidefuision = Guidefusion(c2)
    def forward(self, x):
        ir = x[0]
        vis = x[1]
        # output = self.cmf(x)
        # output = vis
        # output = self.cbam(x)
        # output = torch.cat((ir,vis),dim=1)
        output = torch.add(ir,vis)
        # output = self.guidefuision(x)
        offset = self.conv2_offset(output)
        mask = torch.sigmoid(self.conv2_mask(output))
        vis = self.act2(self.bn2(self.conv2(vis, offset=offset, mask=mask)))
        x = [ir,vis]
        # output = self.cmf(x)
        # output = torch.add(ir,vis)
        return vis
class CMF(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(CMF,self).__init__()
        out_channel = in_channel * ratio
        self.s_weight_classifier = nn.Sequential(
            Convrelu_1(in_channels=out_channel, out_channels=in_channel),
            Conv_3(in_channels= in_channel, out_channels=in_channel),
            nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.x0 = nn.Parameter(torch.tensor(-160.0))  # 偏移量
        self.k = nn.Parameter(torch.tensor(0.01))  # 斜率
    def forward(self, x):
        ir = x[0]
        vis = x[1]
        ill = x[2]
        w_ir = 1 / (1 + torch.exp(-self.k * (ill - self.x0)))
        w_rgb = 1 - w_ir
        concat_feature = torch.concat((vis, ir),dim=1)  # (B, 2C,H,W)
        output = self.s_weight_classifier(concat_feature)
        w_concat = self.avgpool(output)
        w_concat_ir = 1 - w_concat
        # att_r1_visula = w_rgb
        # att_t1_visula = w_ir
        # att_r1_visula = feature_visual(att_r1_visula)
        # att_t1_visula = feature_visual(att_t1_visula)
        # cv2.imwrite(os.path.join('/server08/ljc/OBB/yolov5_obb-master/keshihua/w_ir.jpg'), att_t1_visula)
        # cv2.imwrite(os.path.join('/server08/ljc/OBB/yolov5_obb-master/keshihua/w_vi.jpg'), att_r1_visula)
        output = (w_rgb + w_concat) * vis + ir * (w_ir + w_concat_ir)
        # output = w_concat  * vis + ir * w_concat_ir
        # output = torch.mul(vis , w_vis) + torch.mul(w_ir ,ir)
        return output

class CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c2):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()

    def forward(self, x):
        # x = self.act(self.bn(self.conv(x)))
        ir = x[0]
        vis =x[1]
        ir = self.sa(ir) * ir
        vis = self.sa(vis) * vis
        all = torch.add(ir,vis)
        # vis = vis * self.ca(all) + vis
        # ir = ir * self.ca(all) + ir
        # x = torch.add(ir,vis)
        return all

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


def forward(self, x):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    x = torch.cat([avg_out, max_out], dim=1)
    x = self.conv(x)
    return self.sigmoid(x)
class DSRM(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(DSRM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(3 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.block4 = nn.Sequential(
            BasicConv2d(4 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(torch.cat([x, x1], dim=1))
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))
        return out
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class SIM(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)
    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        # actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = self.bn(normalized * (1 + gamma)) + beta
        return out
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
    def forward(self,x):
        rgb_feature1 = x[0]
        rgb_feature2 = x[1]
        rgb_feature3 = x[2]
        ir_feature1 = x[3]
        ir_feature2 = x[4]
        ir_feature3 = x[5]
class Guidefusion(nn.Module):
    # Standard convolution
    def __init__(self, c2):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Guidefusion, self).__init__()
        self.se = SEAttention(c2 * 2)
    def forward(self, x):
        # x = self.act(self.bn(self.conv(x)))
        ir = x[0]
        vis =x[1]
        x_concat = torch.cat([ir,vis],dim = 1) # n c w h
        x_concat = self.se(x_concat)
        ir_weight , vis_weight = torch.split(x_concat,[ir.size()[1] , vis.size()[1]],dim=1)
        vi_missing = ir * ir_weight
        ir_missing = vis * vis_weight
        output = ir + vis + ir_missing + vi_missing
        return output
class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])




#aligned
class LatentG(nn.Module):
    """ z->w"""

    def __init__(self):
        super(LatentG, self).__init__()
        self.layers = []
        for _ in range(8):
            # fc = nn.Linear(512, 512)
            # act = nn.LeakyReLU()
            # self.layers.append(fc)
            # self.layers.append(act)
            fc = FC(512, 512, 2 ** (0.5), lrmul=0.01, use_wscale=True)
            self.layers.append(fc)
        self.func = nn.Sequential(*self.layers)
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        z = torch.randn(x.size(0), 512, device=x.device, dtype=x.dtype)
        # z = F.normalize(z)
        z = self.pixel_norm(z)
        z = self.func(z)
        return z
class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class RgbIrSTN(nn.Module):
    """
    Spatial Transformer Network module by hmz
    """

    def __init__(self, in_channels, kernel_size=3, use_dropout=False):
        super(RgbIrSTN, self).__init__()
        self._in_ch = in_channels
        self._ksize = kernel_size
        self.dropout = use_dropout

        # localization net
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.bn4 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 6)

        # Initialize the weights/bias with identity transformation
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        one_feat = x[0]
        another_feat = x[1]
        feat = torch.cat((one_feat, another_feat), 1)

        x = F.relu(self.conv1(feat.detach()))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        # x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.adaptive_avg_pool2d(x, 5)

        x = x.view(-1, 32 * 5 * 5)
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x)  # params [Nx6]

        x = x.view(-1, 2, 3)  # change it to the 2x3 matrix
        # print(x.size())
        affine_grid_points = F.affine_grid(x, one_feat.size(),align_corners=True)
        assert (affine_grid_points.size(0) == one_feat.size(
            0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(one_feat, affine_grid_points,align_corners=True)
        # print("rois found to be of size:{}".format(rois.size()))
        return rois
class ACF(nn.Module):
    """
    AdalIN_Concat_Fusion x[0] x[1]-》w style && concat
    """

    def __init__(self, in_channels, dim=1):
        super(ACF, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        # self.fc = nn.Linear(512, self.in_channels * 2)
        self.fc = FC(512, self.in_channels * 2, gain=1.0, use_wscale=True)

    def cal_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x):
        assert (x[0].size()[:2] == x[1].size()[:2])
        rgb_feat = x[1]
        ir_feat = x[0]
        w_style = x[2]
        size = ir_feat.size()
        N, C = size[:2]

        # w_style = F.leaky_relu(self.fc(w_style))
        w_style = self.fc(w_style)
        style_mean = w_style[:, :self.in_channels].view(N, C, 1, 1)
        style_std = w_style[:, self.in_channels:].view(N, C, 1, 1)

        rgb_mean, rgb_std = self.cal_mean_std(rgb_feat)
        ir_mean, ir_std = self.cal_mean_std(ir_feat)

        rgb_feat = (rgb_feat - rgb_mean.expand(size) / rgb_std.expand(size))
        ir_feat = (ir_feat - ir_mean.expand(size)) / ir_std.expand(size)

        rgb_feat = rgb_feat * (style_std.expand(size) + 1.) + style_mean.expand(size)
        ir_feat = ir_feat * (style_std.expand(size) + 1.) + style_mean.expand(size)
        return torch.cat((rgb_feat, ir_feat), self.dim)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class cross_fusion(nn.Module):
    # Standard convolution
    def __init__(self, c2):  # ch_in, ch_out, kernel, stride, padding, groups
        super(cross_fusion, self).__init__()
        self.ca = ChannelAttention(c2 * 2)
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x = self.act(self.bn(self.conv(x)))
        ir = x[0]
        vis =x[1]
        input = torch.cat((ir,vis),dim=1)
        sa_out = self.sa(input)
        ca_out = self.sa(input)
        ir_weight = self.sigmoid(ca_out * sa_out)
        vis_weight = 1 - ir_weight
        new_vis = vis_weight * vis
        new_ir = ir_weight * ir
        output = torch.add (new_ir,new_vis)
        return output
class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out
class Add_CrossPath(nn.Module):
    #  Add two tensors
    def __init__(self, dim, reduction=1, num_heads=8,norm_layer=nn.BatchNorm2d):
        super(Add_CrossPath, self).__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        # output = torch.add(x1,x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        # output = x[0]
        return merge
class Frefusion(nn.Module):
    # Standard convolution
    def __init__(self, c2):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Frefusion, self).__init__()
        self.Frefusion = FreqFusion(hr_channels=c2, lr_channels=c2).to(device=5)
    def forward(self, x):
        ir = x[0]
        vis =x[1]
        _, ir, vis = self.Frefusion(hr_feat=ir, lr_feat=vis)
        output = torch.add(ir,vis)
        return output
class Output(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Output, self).__init__()
        self.arg = arg

    def forward(self, x):
        pred_ir = x[0]
        pred_rgb = x[1]
        w_rgb = x[2]
        # return (pred_ir,pred_rgb)
        return (pred_ir,pred_rgb,w_rgb)
class Feature_cross(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(Feature_cross, self).__init__()
        out_channel = in_channel * ratio
        self.s_weight_classifier = nn.Sequential(
            Convrelu_1(in_channels=out_channel, out_channels=in_channel),
            Conv_3(in_channels=in_channel, out_channels=in_channel),
            nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        vis = x[0]
        ir = x[1]
        # self.vis11(vis)
        # self.vis11(ir)
        concat_feature = torch.concat((vis, ir), dim=1)  # (B, 2C,H,W)
        output = self.s_weight_classifier(concat_feature)
        w_concat = self.avgpool(output)
        w_concat_ir = 1 - w_concat
        vis_missing = w_concat_ir * ir
        ir_missing = w_concat *vis
        return ir_missing, vis_missing
class LJC2(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2*vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock_origin(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)
        self.conv1x1 = nn.Conv2d(d_model*4, d_model, kernel_size=1, stride=1, padding=0)
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.005)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb.shape[0] == ir.shape[0]
        bs, c, h, w = rgb.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb)
        ir_fea = self.avgpool(ir)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.concat((rgb_fea_flat, ir_fea_flat),dim=2) # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs,  self.vert_anchors, self.horz_anchors, 2*self.n_embd)
        x = x.permute(0, 3, 1, 2)  # dim:(B, C, H, W)

        fus_fea = F.interpolate(x, size=([h, w]), mode='bilinear')
        output = torch.cat([rgb,ir,fus_fea],dim=1)
        output = self.conv1x1(output)
        return output
class myTransformerBlock_origin(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention_origin(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()
        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))
        # ill = x[1]
        # x = x[0]
        # bs, nx, c = x.size()
        return x
class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        fea = x[0]
        ill = x[1]
        bs, nx, c = fea.size()
        x = fea + self.sa([self.ln_input(fea),ill])
        x = x + self.mlp(self.ln_output(x))
        # ill = x[1]
        # x = x[0]
        # bs, nx, c = x.size()
        #
        # x = x + self.sa(self.ln_input(x), ill)
        # x = x + self.mlp(self.ln_output(x))
        # x = torch.cat((x.unsqueeze(0), ill.unsqueeze(0)), dim=0)
        return [x,ill]
class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.ill_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        fea = x[0]
        ill = x[1]
        b_s, nq = fea.shape[:2]
        nk = fea.shape[1]
        q = self.que_proj(fea).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(fea).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(fea).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        ill = self.ill_proj(ill).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        v = v * ill
        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)
        # visualize_tensor(att, save_path='./ljc/att.jpg')
        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out
class SelfAttention_origin(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention_origin, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        # self.ill_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        # ill = self.ill_proj(ill).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        # v = v * ill
        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)
        # visualize_tensor(att, save_path='./ljc/att.jpg')
        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out
class Output1(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Output1, self).__init__()
        self.arg = arg

    def forward(self, x):
        pred_ir = x[0]
        pred_rgb = x[1]
        # return (pred_ir,pred_rgb)
        return pred_ir,pred_rgb
class Illumination_classifier(nn.Module):
    def __init__(self, input_channels, init_weights=True):
        super(Illumination_classifier, self).__init__()
        self.conv1 = reflect_conv(in_channels=3, out_channels=16)
        self.conv2 = reflect_conv(in_channels=16, out_channels=32)
        self.conv3 = reflect_conv(in_channels=32, out_channels=64)
        self.conv4 = reflect_conv(in_channels=64, out_channels=128)
        self.conv5 = reflect_conv(in_channels=128, out_channels=256)
        self.conv1_x1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_x2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_x3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.linear1 = nn.Linear(in_features=128, out_features=128)
        self.linear2 = nn.Linear(in_features=1, out_features=2)

    def forward(self, x):
        # x = x * 255
        # img = x.squeeze(0)
        # img = img.detach().cpu().numpy().transpose(1, 2, 0)
        # img = np.clip(img, 0, 255).astype(np.uint8)
        # # 保存图像
        # f = '/server08/ljc/OBB/yolov5_obb-master/test.jpg'
        # Image.fromarray(img).save(f)
        activate = nn.LeakyReLU(inplace=True)
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x_1 = activate(self.conv3(x))
        x_2 = activate(self.conv4(x_1))
        x_3 = activate(self.conv5(x_2))
        ill_1 = self.conv1_x1(x_1)
        ill_2 = self.conv1_x2(x_2)
        ill_3 = self.conv1_x3(x_3)
        x = nn.AdaptiveAvgPool2d(1)(ill_1)
        x = x.view(x.size(0), -1)
        # x = self.linear1(x)
        # x = activate(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)  # 设置ReLU激活函数，过滤负值
        # x = nn.Sigmoid()(x)
        # x = nn.ReLU(inplace=True)(x)
        return ill_1
class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

#ICAFusion
class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1,
                 attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.crosstransformer = nn.Sequential(
            *[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in
              range(n_layer)])

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        # new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        # new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        # ------------------------- feature visulization -----------------------#
        # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
        # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
        # fea_ir_new = torch.mean(new_ir_fea, dim=1)
        # fea_new = torch.mean(new_fea, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
        # -----------------------------------------------------------------------------#

        return new_fea
class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y
class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out
class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention_ICA(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

            # without Learnable Coefficient
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.LN2(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.LN2(ir_att_out))

        return [rgb_fea_flat, ir_fea_flat]
class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out
class CrossAttention_ICA(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention_ICA, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)

        return [out_vis, out_ir]

class LJC3(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2*vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)
        self.conv1x1 = nn.Conv2d(d_model*4, d_model, kernel_size=1, stride=1, padding=0)
        self.x0 = nn.Parameter(torch.tensor(-160.0))  # 偏移量
        self.k = nn.Parameter(torch.tensor(0.01))  # 斜率
        self.conv1x1_ill = nn.Conv2d(1, 1, kernel_size=1)  # 输入通道数为1，输出通道数为1
        self.sigmoid = nn.Sigmoid()  # 用于将输出值限制在0到1之间
        self.sa = SpatialAttention()
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        ir = x[0]
        rgb = x[1]
        ill = x[2]
        output_size = ir.size()[2:]
        ill = 1 / (1 + torch.exp(-self.k * (ill - self.x0)))
        ill = F.adaptive_avg_pool2d(ill, output_size)  # 使用自适应平均池化调整大小
        ill = self.conv1x1_ill(ill)
        ill = self.sigmoid(ill)
        w_ir = ill
        w_rgb = 1 - ill
        # visualize_tensor(w_ir, save_path='./ljc/w_ir.jpg')
        # visualize_tensor(w_rgb, save_path='./ljc/w_rgb.jpg')
        # print(self.k)
        assert rgb.shape[0] == ir.shape[0]
        bs, c, h, w = rgb.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb)
        ir_fea = self.avgpool(ir)
        w_rgb = self.avgpool(w_rgb)
        w_ir = self.avgpool(w_ir)
        w_rgb = w_rgb.expand(-1,c, -1, -1)
        w_ir = w_ir.expand(-1,c,-1,-1)
        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        w_rgb_fea_flat = w_rgb.view(bs, c, -1)  # flatten the feature
        w_ir_fea_flat = w_ir.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.concat((rgb_fea_flat, ir_fea_flat), dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)
        token_ill = torch.concat((w_rgb_fea_flat, w_ir_fea_flat), dim=2)  # concat
        token_ill = token_ill.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        fea = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        ill = self.drop(self.pos_emb + token_ill)
        x = self.trans_blocks([fea,ill])  # dim:(B, 2*H*W, C)
        x = x[0]
        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, self.vert_anchors, self.horz_anchors, 2 * self.n_embd)
        x = x.permute(0, 3, 1, 2)  # dim:(B, C, H, W)

        fus_fea = F.interpolate(x, size=([h, w]), mode='bilinear')
        output = torch.cat([rgb, ir, fus_fea], dim=1)
        # visualize_tensor(rgb, save_path='./ljc/rgb.jpg')
        # visualize_tensor(ir, save_path='./ljc/ir.jpg')
        # visualize_tensor(fus_fea, save_path='./ljc/fus_fea.jpg')
        output = self.conv1x1(output)
        # visualize_tensor(output, save_path='./ljc/output.jpg')
        return output


class MF(nn.Module):
    def __init__(self, channels):
        super(MF, self).__init__()
        self.mask_map_r = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)

        # 修改bottleneck，添加下采样
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1, bias=False),
            nn.Conv2d(16, 16, 3, 2, 1, bias=False)  # stride=2下采样
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(channels, 48, 3, 1, 1, bias=False),
            nn.Conv2d(48, 48, 3, 2, 1, bias=False)  # stride=2下采样
        )
        self.se = SE_Block(64, 16)

    def forward(self, x):
        x_left_ori, x_right_ori = x[0], x[1]
        b, c, h, w = x_left_ori.shape

        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left).repeat(1, 3, 1, 1), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right).repeat(1, 3, 1, 1), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)  # 输出: [8, 16, 320, 320]
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # 输出: [8, 48, 320, 320]
        out = self.se(torch.cat([out_RGB, out_IR], 1))  # 输出: [8, 64, 320, 320]

        return out
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上

