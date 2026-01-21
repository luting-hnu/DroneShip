import math
from pathlib import Path
from torch import nn
import torch
from models.common import reflect_conv
import matplotlib.pyplot as plt
def visualize_tensor(x, save_path):
    # Ensure the save directory exists
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    # Extract tensor shape
    batch, channels, height, width = x.shape
    if height > 1 and width > 1:
        # Sum across channels to visualize as a single image
        img = x[0].cpu().transpose(0, 1).sum(1).detach().numpy()
        # Normalize the image to [0, 1] for better visualization
        img = (img - img.min()) / (img.max() - img.min())
        # Save the image
        plt.imsave(save_path, img, cmap='viridis')  # Use a colormap for better visualization
class Illumination_classifier(nn.Module):
    def __init__(self, input_channels, init_weights=True):
        super(Illumination_classifier, self).__init__()
        self.conv1 = reflect_conv(in_channels=input_channels, out_channels=16)
        self.conv2 = reflect_conv(in_channels=16, out_channels=32)
        self.conv3 = reflect_conv(in_channels=32, out_channels=64)
        self.conv4 = reflect_conv(in_channels=64, out_channels=128)
        self.conv5 = reflect_conv(in_channels=128, out_channels=256)
        self.conv1_x1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_x2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_x3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.linear1 = nn.Linear(in_features=128, out_features=128)
        self.linear2 = nn.Linear(in_features=1, out_features=2)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化权重

        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        activate = nn.LeakyReLU(inplace=True)
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x_1 = activate(self.conv3(x))
        x_2 = activate(self.conv4(x_1))
        x_3 = activate(self.conv5(x_2))
        ill_1 = self.conv1_x1(x_1)
        ill_2 = self.conv1_x2(x_2)
        ill_3 = self.conv1_x3(x_3)
        x0 = nn.Parameter(torch.tensor(-160.0))  # 偏移量
        k = nn.Parameter(torch.tensor(0.01))  # 斜率
        ill = 1 / (1 + torch.exp(-k * (ill_1 - x0)))

        x = nn.AdaptiveAvgPool2d(1)(ill_1)
        x = x.view(x.size(0), -1)
        # x = self.linear1(x)
        # x = activate(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)  # 设置ReLU激活函数，过滤负值
        # x = nn.Sigmoid()(x)
        # x = nn.ReLU(inplace=True)(x)
        return x


