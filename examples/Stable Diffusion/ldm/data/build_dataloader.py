import torch  # 导入 PyTorch 库，用于张量操作和数据加载器
# from image_synthesis.data.base_dataset import ConcatDatasetWithIndex as ConcatDataset  # 备选：自定义带索引的拼接数据集实现（当前未使用）
from torch.utils.data import ConcatDataset  # 使用 PyTorch 的 ConcatDataset 将多个数据集拼接为一个
from ldm.util import instantiate_from_config  # 从配置字典实例化对象的工具函数


def build_dataloader(config, args=None, return_dataset=False):  # 构建训练与验证数据加载器，支持返回数据集对象
    dataset_cfg = config['dataloader']  # 读取 dataloader 配置部分
    train_dataset = []  # 收集多个训练数据集的列表
    if 'train_datasets' in dataset_cfg:  # 如果配置中包含训练数据集定义
        for ds_cfg in dataset_cfg['train_datasets']:  # 遍历每个训练数据集的配置
            ds_cfg['params']['data_root'] = dataset_cfg.get('data_root', '')  # 将全局 data_root 注入到每个数据集的参数中
            ds = instantiate_from_config(ds_cfg)  # 根据配置实例化具体数据集对象
            train_dataset.append(ds)  # 添加到训练数据集列表
        if len(train_dataset) > 1:  # 如果有多个训练数据集
            train_dataset = ConcatDataset(train_dataset)  # 使用 ConcatDataset 将其拼接为一个大数据集
        else:
            train_dataset = train_dataset[0]  # 仅一个数据集时直接使用该对象
    
    val_dataset = []  # 收集多个验证数据集的列表
    if 'validation_datasets' in dataset_cfg:  # 如果配置中包含验证数据集定义
        for ds_cfg in dataset_cfg['validation_datasets']:  # 遍历每个验证数据集的配置
            ds_cfg['params']['data_root'] = dataset_cfg.get('data_root', '')  # 注入全局 data_root
            ds = instantiate_from_config(ds_cfg)  # 实例化数据集
            val_dataset.append(ds)  # 添加到验证数据集列表
        if len(val_dataset) > 1:  # 多个验证数据集
            val_dataset = ConcatDataset(val_dataset)  # 拼接为一个验证数据集
        else:
            val_dataset = val_dataset[0]  # 仅一个时直接取出
    
    # 以下为分布式采样器的示例代码，当前注释掉：可用于多进程训练以保障数据划分与打乱
    # if args is not None and args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    #     train_iters = len(train_sampler) // dataset_cfg['batch_size']
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    #     val_iters = len(val_sampler) // dataset_cfg['batch_size']
    # else:
    #     train_sampler = None
    #     val_sampler = None
    #     train_iters = len(train_dataset) // dataset_cfg['batch_size']
    #     val_iters = len(val_dataset) // dataset_cfg['batch_size']
    
    train_sampler = None  # 当前不启用分布式采样器
    val_sampler = None  # 验证集同样不使用采样器
    train_iters = len(train_dataset) // dataset_cfg['batch_size']  # 计算训练迭代次数：样本数除以 batch_size
    val_iters = len(val_dataset) // dataset_cfg['batch_size']  # 计算验证迭代次数

    # 动态 num_workers 的示例代码，已注释：可根据 batch_size 调整 IO 并发
    # if args is not None and not args.debug:
    #     num_workers = max(2*dataset_cfg['batch_size'], dataset_cfg['num_workers'])
    #     num_workers = min(64, num_workers)
    # else:
    #     num_workers = dataset_cfg['num_workers']
    num_workers = dataset_cfg['num_workers']  # 使用配置中固定的工作线程数
    train_loader = None  # 默认训练数据加载器为空
    val_loader = None  # 默认验证数据加载器为空
    if len(train_dataset) > 0:  # 若训练数据集非空
        train_loader = torch.utils.data.DataLoader(train_dataset,  # 构建训练 DataLoader
                                                batch_size=dataset_cfg['batch_size'],  # 批大小
                                                shuffle=(train_sampler is None),  # 若无采样器则打乱数据
                                                num_workers=num_workers,  # 数据加载并发线程数
                                                pin_memory=True,  # 将张量固定到页锁内存，加速 GPU 传输
                                                sampler=train_sampler,  # 采样器（此处为 None）
                                                drop_last=True,  # 丢弃不满一个 batch 的最后一批，保证 batch 大小一致
                                                persistent_workers=True)  # 持久化 worker，提升多轮迭代效率

    if len(val_dataset) > 0:  # 若验证数据集非空
        val_loader = torch.utils.data.DataLoader(val_dataset,  # 构建验证 DataLoader
                                                batch_size=dataset_cfg['batch_size'],  # 批大小
                                                shuffle=False, #(val_sampler is None),  # 验证集通常不打乱
                                                num_workers=num_workers,  # 并发线程数
                                                pin_memory=True,  # 页锁内存
                                                sampler=val_sampler,  # 验证采样器（此处为 None）
                                                drop_last=True,  # 丢弃最后不满批次
                                                persistent_workers=True)  # 持久化 worker

    dataload_info = {  # 汇总数据加载器与迭代信息
        'train_loader': train_loader,
        'validation_loader': val_loader,
        'train_iterations': train_iters,
        'validation_iterations': val_iters
    }
    
    if return_dataset:  # 可选地返回数据集对象，便于外部访问和调试
        dataload_info['train_dataset'] = train_dataset
        dataload_info['validation_dataset'] = val_dataset

    return dataload_info  # 返回包含加载器与迭代计数的字典