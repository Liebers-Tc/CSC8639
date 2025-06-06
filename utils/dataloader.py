from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


class FoodSegDataset(Dataset):
    """
    食物图像分割数据集类，适用于训练 / 验证 / 测试集的统一加载。
    - root_dir: 数据集的根目录，结构应为：
        root_dir/
            train/
                images/
                masks/
            val/
                images/
                masks/
            test/
                images/
                masks/
    - dataset: 指定子集类型，字符串，可选值为 "train"、"val" 或 "test"
    """

    # 将 PIL 格式的 mask 图像转换为 Long 类型的 Tensor，并移除可能多余的通道维度
    @staticmethod
    def _mask_to_tensor(m):
        return torch.as_tensor(np.array(m), dtype=torch.long).squeeze()
    
    def __init__(self, root_dir, dataset, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.image_dir = Path(root_dir) / dataset / 'images'
        self.mask_dir = Path(root_dir) / dataset / 'masks'
        self.image_list = sorted(list(self.image_dir.glob('*.jpg')))

        # 图像预处理：转为 Tensor 并标准化
        self.image_transform = T.Compose([
            # T.Resize((256, 256), interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean, std)
            ])
        
        # 掩码预处理：读取为 Long 类型张量（不做归一化或缩放）
        self.mask_transform = T.Compose([
            # T.Resize((256, 256), interpolation=Image.NEAREST),
            T.Lambda(self._mask_to_tensor)
            ])
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 获取单张图像路径
        image_path = self.image_list[idx]
        # 获取单张掩码路径（将后缀 .jpg 替换为 .png）
        mask_path = self.mask_dir / image_path.with_suffix('.png').name

        # 加载图像和掩码并应用预处理
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask