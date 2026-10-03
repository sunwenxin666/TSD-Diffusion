from abc import abstractmethod  # 导入抽象方法装饰器，用于定义必须由子类实现的方法
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset  # 从 PyTorch 中导入数据集相关基类，包括可迭代数据集


class Txt2ImgIterableBaseDataset(IterableDataset):  # 定义一个用于文本到图像任务的可迭代数据集基类，继承 IterableDataset
    '''
    Define an interface to make the IterableDatasets for text2img data chainable  # 英文原始注释：定义接口以使文本到图像的可迭代数据集可链式组合
    该基类用于为文本到图像（text2img）任务提供统一的可迭代数据集接口，便于多个数据集串联或拼接使用。  # 中文解释
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):  # 初始化：num_records 为样本数量；valid_ids 表示有效样本 ID 列表；size 表示图像目标尺寸
        super().__init__()  # 调用父类 IterableDataset 的初始化
        self.num_records = num_records  # 保存样本总数，用于 __len__ 返回
        self.valid_ids = valid_ids  # 保存有效 ID 列表，通常用于过滤或采样
        self.sample_ids = valid_ids  # 将 sample_ids 指向 valid_ids，便于迭代时使用
        self.size = size  # 保存图像处理的目标尺寸（如裁剪/缩放的目标大小）

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')  # 打印数据集信息，包含类名和样本数量

    def __len__(self):  # 定义数据集长度获取方法
        return self.num_records  # 返回样本数量，供数据加载器或日志统计使用

    @abstractmethod  # 抽象方法装饰器，要求子类必须实现该方法
    def __iter__(self):  # 迭代器方法，返回一个迭代器以按样本顺序产出数据
        pass  # 基类不实现具体迭代逻辑，留给子类按任务需求重写