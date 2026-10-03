import albumentations  # 图像增强库，用于随机裁剪、中心裁剪等变换
import random  # 随机数库，用于产生随机偏移和尺寸
import numpy as np  # 数值计算库，用于数组操作
from PIL import Image  # 图像处理库，用于读写和缩放裁剪图像
import cv2  # OpenCV 库（本文件未直接使用，但常用于图像处理）
from io import BytesIO  # 字节缓冲区（当前未使用）
from torchvision import transforms as trans  # torchvision 的变换接口（当前未使用）

class DalleTransformerPreprocessor(object):  # 仿照 DALL·E 的预处理器：训练阶段随机裁剪，验证阶段中心裁剪
    def __init__(self, 
                 size=256,  # 目标裁剪尺寸（正方形边长）
                 phase='train',  # 阶段：'train' 或 'val'
                 additional_targets=None):  # albumentations 的额外目标（如 mask），默认无
    
        self.size = size  # 保存目标尺寸
        self.phase = phase  # 保存当前阶段
        # ddc: following dalle to use randomcrop  # 说明：训练阶段使用随机裁剪策略
        self.train_preprocessor = albumentations.Compose([albumentations.RandomCrop(height=size, width=size)],  # 定义训练预处理：随机裁剪到 size
                                                   additional_targets=additional_targets)
        self.val_preprocessor = albumentations.Compose([albumentations.CenterCrop(height=size, width=size)],  # 定义验证预处理：中心裁剪到 size
                                                   additional_targets=additional_targets)                                                   


    def __call__(self, image, **kargs):  # 使实例可调用，输入 PIL.Image 或 numpy 数组，输出裁剪后的图像
        """
        image: PIL.Image  # 期望输入类型为 PIL.Image（若为 numpy，将转换）
        """
        if isinstance(image, np.ndarray):  # 若是 numpy 数组，将其转换为 PIL.Image
            image = Image.fromarray(image.astype(np.uint8)) 

        w, h = image.size  # 原始图像宽高
        s_min = min(h, w)  # 短边长度，用于裁剪为正方形

        if self.phase == 'train':  # 训练阶段：仿 DALL·E 的随机偏移 + 随机缩放 + 随机裁剪
            off_h = int(random.uniform(3*(h-s_min)//8, max(3*(h-s_min)//8+1, 5*(h-s_min)//8)))  # 随机选择竖直偏移
            off_w = int(random.uniform(3*(w-s_min)//8, max(3*(w-s_min)//8+1, 5*(w-s_min)//8)))  # 随机选择水平偏移
            # import pdb; pdb.set_trace()  # 调试语句（注释掉）
            image = image.crop((off_w, off_h, off_w + s_min, off_h + s_min))  # 按偏移裁剪出 s_min×s_min 的区域

            # resize image  # 重设尺寸到随机 t（不小于目标 size）
            t_max = min(s_min, round(9/8*self.size))  # 缩放上限，取 min(s_min, 9/8*size)
            t_max = max(t_max, self.size)  # 确保上限不小于 size
            t = int(random.uniform(self.size, t_max+1))  # 在 [size, t_max] 之间随机选取整数尺寸 t
            image = image.resize((t, t))  # 将正方形缩放到 t×t
            image = np.array(image).astype(np.uint8)  # 转为 numpy 数组
            image = self.train_preprocessor(image=image)   #randomcrop (size,size)  # 随机裁剪到 size×size
        else:  # 验证阶段：按短边等比缩放到 size，随后中心裁剪
            if w < h:
                w_ = self.size  # 将宽缩放到 size
                h_ = int(h * w_/w)  # 高按比例缩放
            else:
                h_ = self.size  # 将高缩放到 size
                w_ = int(w * h_/h)  # 宽按比例缩放
            image = image.resize((w_, h_))  # 等比缩放
            image = np.array(image).astype(np.uint8)  # 转为 numpy 数组
            image = self.val_preprocessor(image=image)  # 中心裁剪到 size
        return image  # 返回处理结果（albumentations 返回字典或数组，依版本而定）


class ImageNetTransformerPreprocessor(object):  # ImageNet 任务的预处理器：训练随机裁剪，验证中心裁剪
    def __init__(self, 
                 size=256,  # 目标尺寸
                 phase='train',  # 阶段
                 additional_targets=None):  # 额外目标
    
        self.size = size  # 保存尺寸
        self.phase = phase  # 保存阶段
        # ddc: following dalle to use randomcrop  # 说明：训练阶段使用随机裁剪
        self.train_preprocessor = albumentations.Compose([albumentations.RandomCrop(height=size, width=size)],  # 随机裁剪
                                                   additional_targets=additional_targets)
        self.val_preprocessor = albumentations.Compose([albumentations.CenterCrop(height=size, width=size)],  # 中心裁剪
                                                   additional_targets=additional_targets)                                                   


    def __call__(self, image, **kargs):  # 可调用接口，输入图像，输出处理后结果
        """
        image: PIL.Image  # 输入图像类型
        """
        if isinstance(image, np.ndarray):  # 若为 numpy 数组，转换为 PIL.Image
            image = Image.fromarray(image.astype(np.uint8)) 

        w, h = image.size  # 获取宽高
        s_min = min(h, w)  # 获取短边

        if self.phase == 'train':  # 训练阶段：先按短边等比缩放到 size，再随机裁剪到 size
            if w < h:
                w_ = self.size  # 宽缩放到 size
                h_ = int(h * w_/w)  # 高按比例缩放
            else:
                h_ = self.size  # 高缩放到 size
                w_ = int(w * h_/h)  # 宽按比例缩放
            image = image.resize((w_, h_))  # 进行缩放
            image = np.array(image).astype(np.uint8)  # 转为 numpy 数组
            image = self.train_preprocessor(image=image)  # 随机裁剪到 size
        else:  # 验证阶段：等比缩放后中心裁剪
            if w < h:
                w_ = self.size  # 宽缩放到 size
                h_ = int(h * w_/w)  # 高按比例缩放
            else:
                h_ = self.size  # 高缩放到 size
                w_ = int(w * h_/h)  # 宽按比例缩放
            image = image.resize((w_, h_))  # 等比缩放
            image = np.array(image).astype(np.uint8)  # 转 numpy 数组
            image = self.val_preprocessor(image=image)  # 中心裁剪到 size
        return image  # 返回结果

