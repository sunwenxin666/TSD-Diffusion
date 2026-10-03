from torch.utils.data import Dataset  # 从 PyTorch 导入数据集基类，用于定义自定义数据集
import numpy as np  # 导入 NumPy，用于数组和数值处理
import io  # 导入 io（当前未使用），通常用于字节流读写
from PIL import Image  # 导入 PIL 的 Image 类，用于图像读写与处理
import os  # 导入 os，用于文件路径拼接和文件系统操作
import json  # 导入 json，用于读取 COCO 的标注文件
import random  # 导入 random（当前未使用），通常用于随机采样
# from image_synthesis.utils.misc import instantiate_from_config  # 旧路径的配置实例化方法，已被替换
from ldm.util import instantiate_from_config  # 从项目工具模块导入配置实例化函数，根据配置创建预处理器/组件


def load_img(filepath):  # 定义图像加载函数，入参为文件路径
    img = Image.open(filepath).convert('RGB')  # 使用 PIL 打开图像并转换为 RGB 格式，确保有 3 个通道
    return img  # 返回 PIL Image 对象


class CocoDataset(Dataset):  # 定义 COCO 数据集类，继承自 PyTorch 的 Dataset
    def __init__(self, data_root, phase = 'train', im_preprocessor_config=None):  # 初始化函数，接受数据根目录、阶段名、图像预处理器配置
        self.transform = instantiate_from_config(im_preprocessor_config)  # 根据配置创建图像预处理器（Albumentations/自定义等），返回一个可调用对象
        self.root = data_root  # 记录数据根目录路径
        # caption_file = os.path.join(data_root, "annotations", caption_file)  # 原始用法：从 annotations/ 目录读取指定的 caption 文件
        caption_file = os.path.join(data_root, "captions_val2014_30K_samples.json")  # 当前固定读取一个包含 30K 验证样本的 caption JSON 文件

        self.json_file = json.load(open(caption_file, 'r'))  # 打开并解析 JSON 文件，得到包含注释信息的字典
        print("length of the dataset is ")  # 打印提示信息（数据集长度）
        print(len(self.json_file['annotations']))  # 打印注释条目数（作为数据集样本数）

        self.num = len(self.json_file['annotations'])  # 记录样本总数
        # self.image_prename = "COCO_" + phase + "2014_"  # 原始命名规则：根据阶段拼前缀，例如 COCO_train2014_
        self.image_prename = 'COCO_val2014_'  # 现版本固定为验证集前缀 COCO_val2014_
        # self.folder_path = os.path.join(data_root, phase+'2014', phase+'2014')  # 原始路径两级包含阶段名，例如 train2014/train2014
        self.folder_path = os.path.join(data_root, 'val2014')  # 现版本固定图像目录为 val2014
        # self.folder_path = os.path.join(data_root, 'val')  # 另一种可能的验证目录结构（当前未使用）
 
    def __len__(self):  # 返回数据集样本数的魔术方法
        return self.num  # 返回记录的样本总数
 
    def __getitem__(self, index):  # 根据索引返回一个样本的魔术方法
        this_item = self.json_file['annotations'][index]  # 从 JSON 的 annotations 列表中取出对应索引的注释条目
        caption = this_item['caption'].lower()  # 读取文本描述并转换为小写，便于统一处理
        image_name = str(this_item['image_id']).zfill(12)  # 将 image_id 转为字符串并左侧补零到 12 位，匹配 COCO 命名规则
        image_path = os.path.join(self.folder_path, self.image_prename+image_name+'.jpg')  # 拼接图像文件路径，如 val2014/COCO_val2014_000000000001.jpg
        image = load_img(image_path)  # 加载图像（PIL Image）
        image = np.array(image).astype(np.uint8)  # 转为 NumPy 数组，类型为 uint8（0-255），便于预处理
        image = self.transform(image = image)['image']  # 调用预处理器对图像执行变换，约定返回字典，取出处理后的 'image'
        data = {  # 组织输出样本字典
                'image': np.transpose(image.astype(np.float32), (2, 0, 1)),  # 将图像转为 float32 并从 HWC 转为 CHW 通道顺序，符合 PyTorch 习惯
                'text': caption,  # 文本描述（caption）
        }
        
        return data  # 返回包含图像与文本的样本字典
