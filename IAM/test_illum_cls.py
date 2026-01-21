import torch
from torchvision import transforms
from PIL import Image
from models.cls_model import Illumination_classifier


if __name__ == '__main__':
    # 设置指定 GPU（第 7 张卡）
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    # 加载图像
    image_path = ''
    image = Image.open(image_path).convert('RGB')

    # 定义图像预处理操作
    transform = transforms.Compose([
        transforms.Resize((406, 519)),
        transforms.ToTensor(),
    ])

    # 对图像进行预处理
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 加载预训练的分类模型
    cls_model = Illumination_classifier(input_channels=3)
    cls_model.load_state_dict(torch.load('', map_location=device))
    cls_model = cls_model.to(device)
    cls_model.eval()

    # 推理
    with torch.no_grad():
        output = cls_model(image_tensor)
        print(output)
        _, predicted = torch.max(output, 1)
        print(predicted)

    # 输出预测结果
    if predicted.item() == 0:
        print("预测结果：白天")
    else:
        print("预测结果：黑夜")