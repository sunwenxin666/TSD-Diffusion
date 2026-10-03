import torch  # 导入 PyTorch，用于张量计算与神经网络构建
import pytorch_lightning as pl  # 导入 PyTorch Lightning，封装训练循环与模块结构
import torch.nn.functional as F  # 导入函数式 API，如卷积、插值等通用操作
from contextlib import contextmanager  # 导入上下文管理器，用于临时切换 EMA 权重等

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer  # 从 taming 模块导入向量量化器实现（VQ-VAE 核心组件）

from ldm.modules.diffusionmodules.model import Encoder, Decoder  # 导入编码器与解码器，用于图像到潜变量以及潜变量到图像的转换
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution  # 导入对角高斯分布，用于变分后验的参数化

from ldm.util import instantiate_from_config  # 从配置中实例化模块（工厂函数）


class VQModel(pl.LightningModule):  # 定义基于 VQ-VAE 的模型（含向量量化与重建）
    def __init__(self,
                 ddconfig,  # 编解码器的超参数字典（如通道数、层数、是否双通道等）
                 lossconfig,  # 损失函数配置（含重建与对抗等）
                 n_embed,  # 码本中嵌入向量的数量（字典大小）
                 embed_dim,  # 每个嵌入向量的维度
                 ckpt_path=None,  # 可选：加载预训练检查点路径
                 ignore_keys=[],  # 加载时忽略的参数前缀列表
                 image_key="image",  # 数据 batch 中图像字段的键名
                 colorize_nlabels=None,  # 可选：用于语义分割多通道可视化的标签数
                 monitor=None,  # 可选：监控指标的键名（如用于早停）
                 batch_resize_range=None,  # 可选：训练中动态调整输入分辨率范围（避免显存溢出）
                 scheduler_config=None,  # 可选：学习率调度器配置
                 lr_g_factor=1.0,  # 生成器（编码器/解码器）学习率的缩放因子
                 remap=None,  # 可选：码本重映射策略
                 sane_index_shape=False, # 若为 True，量化器返回 bhw 维度索引（更直观）
                 use_ema=False  # 是否启用 EMA（指数滑动平均）权重
                 ):
        super().__init__()  # 调用父类初始化（LightningModule）
        self.embed_dim = embed_dim  # 保存嵌入维度
        self.n_embed = n_embed  # 保存码本大小
        self.image_key = image_key  # 保存图像键名
        self.encoder = Encoder(**ddconfig)  # 根据配置实例化编码器
        self.decoder = Decoder(**ddconfig)  # 根据配置实例化解码器
        self.loss = instantiate_from_config(lossconfig)  # 从配置实例化损失模块
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,  # 量化器，beta 为承诺损失系数
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)  # z -> embed_dim 的1x1卷积，匹配码本维度
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)  # embed_dim -> z 的1x1卷积，解码前恢复通道
        if colorize_nlabels is not None:  # 若设置可视化标签数
            assert type(colorize_nlabels)==int  # 必须为整数
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))  # 注册随机投影用于可视化（3通道）
        if monitor is not None:  # 若设置监控键
            self.monitor = monitor  # 保存监控键名
        self.batch_resize_range = batch_resize_range  # 保存动态分辨率范围
        if self.batch_resize_range is not None:  # 若启用动态分辨率
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")  # 打印提示

        self.use_ema = use_ema  # 是否使用 EMA
        if self.use_ema:  # 启用 EMA 则创建跟踪器
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:  # 若提供检查点路径则加载权重
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config  # 保存调度器配置
        self.lr_g_factor = lr_g_factor  # 保存生成器学习率因子

    @contextmanager  # 上下文管理器装饰器：进入时切换到 EMA 权重，退出时恢复训练权重
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())  # 暂存当前权重
            self.model_ema.copy_to(self)  # 覆盖为 EMA 权重
            if context is not None:
                print(f"{context}: Switched to EMA weights")  # 打印提示
        try:
            yield None  # 执行上下文内部代码
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())  # 恢复训练权重
                if context is not None:
                    print(f"{context}: Restored training weights")  # 打印提示

    def init_from_ckpt(self, path, ignore_keys=list()):  # 从检查点加载权重，并忽略部分键
        sd = torch.load(path, map_location="cpu")["state_dict"]  # 加载 state_dict
        keys = list(sd.keys())  # 获取所有键
        for k in keys:
            for ik in ignore_keys:  # 遍历忽略前缀
                if k.startswith(ik):  # 若匹配则删除该键
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)  # 非严格加载，返回缺失与意外键
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")  # 打印恢复信息
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")  # 打印缺失键
            print(f"Unexpected Keys: {unexpected}")  # 打印意外键

    def on_train_batch_end(self, *args, **kwargs):  # 每个训练 batch 结束时回调
        if self.use_ema:
            self.model_ema(self)  # 更新 EMA（滑动平均）

    def encode(self, x):  # 编码并量化：返回量化结果、嵌入损失、信息字典
        h = self.encoder(x)  # 通过编码器得到潜表示 h
        h = self.quant_conv(h)  # 通道映射到嵌入维度
        quant, emb_loss, info = self.quantize(h)  # 向量量化，得到量化张量与损失
        return quant, emb_loss, info  # 返回结果

    def encode_to_prequant(self, x):  # 编码到量化前表示（不做码本匹配）
        h = self.encoder(x)  # 编码图像
        h = self.quant_conv(h)  # 通道映射到嵌入维度
        return h  # 返回量化前张量

    def decode(self, quant):  # 解码量化表示为图像
        quant = self.post_quant_conv(quant)  # 量化后通道映射回 z_channels
        dec = self.decoder(quant)  # 通过解码器还原图像
        return dec  # 返回重建图像

    def decode_code(self, code_b):  # 根据码本索引解码（跳过编码器）
        quant_b = self.quantize.embed_code(code_b)  # 将索引转为嵌入张量
        dec = self.decode(quant_b)  # 解码得到图像
        return dec  # 返回图像

    def forward(self, input, return_pred_indices=False):  # 前向：返回重建与量化损失，可选返回码本索引
        quant, diff, (_,_,ind) = self.encode(input)  # 编码并量化，diff 为嵌入损失，ind 为码本索引
        dec = self.decode(quant)  # 解码得到重建图像
        if return_pred_indices:  # 若需要返回码本索引
            return dec, diff, ind  # 返回重建、损失与索引
        return dec, diff  # 否则返回重建和损失

    def get_input(self, batch, k):  # 从 batch 取出图像，并调整维度顺序与类型
        x = batch[k]  # 取出图像字段
        if len(x.shape) == 3:  # 若为 HWC 则补充通道维
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()  # 转为 NCHW 与 float
        if self.batch_resize_range is not None:  # 若启用动态分辨率
            lower_size = self.batch_resize_range[0]  # 下限尺寸
            upper_size = self.batch_resize_range[1]  # 上限尺寸
            if self.global_step <= 4:  # 前几个 step 使用最大尺寸，避免显存突增
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))  # 随机选择尺寸（步长16）
            if new_resize != x.shape[2]:  # 若高度变化则插值到新尺寸（保持方形）
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()  # 断开梯度，避免插值参与反传
        return x  # 返回预处理后的输入

    def training_step(self, batch, batch_idx, optimizer_idx):  # Lightning 训练步骤，支持多优化器
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)  # 获取输入图像
        xrec, qloss, ind = self(x, return_pred_indices=True)  # 前向，得到重建、量化损失与索引

        if optimizer_idx == 0:  # 优化器 0：自编码器路径（重建/量化/感知等）
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)  # 计算 AE 损失与日志

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)  # 记录日志
            return aeloss  # 返回损失

        if optimizer_idx == 1:  # 优化器 1：判别器路径（GAN 对抗）
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")  # 计算判别器损失
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)  # 记录日志
            return discloss  # 返回判别器损失

    def validation_step(self, batch, batch_idx):  # 验证步骤：同时记录 EMA 版本指标
        log_dict = self._validation_step(batch, batch_idx)  # 正常权重下的验证
        with self.ema_scope():  # 切换到 EMA 权重
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")  # EMA 权重的验证
        return log_dict  # 返回正常权重日志（EMA 日志已内部记录）

    def _validation_step(self, batch, batch_idx, suffix=""):  # 具体验证逻辑，可指定后缀区分 EMA
        x = self.get_input(batch, self.image_key)  # 获取输入
        xrec, qloss, ind = self(x, return_pred_indices=True)  # 前向
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )  # 计算 AE 损失与日志

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )  # 计算判别器损失与日志
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]  # 取重建损失
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)  # 记录到进度条与日志
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)  # 记录 AE 损失
        if version.parse(pl.__version__) >= version.parse('1.4.0'):  # 若 Lightning 版本 >= 1.4.0
            del log_dict_ae[f"val{suffix}/rec_loss"]  # 删除重复项避免冲突
        self.log_dict(log_dict_ae)  # 记录 AE 日志字典
        self.log_dict(log_dict_disc)  # 记录判别器日志字典
        return self.log_dict  # 返回日志字典（Lightning 要求返回映射）

    def configure_optimizers(self):  # 配置优化器与调度器
        lr_d = self.learning_rate  # 判别器学习率
        lr_g = self.lr_g_factor*self.learning_rate  # 生成器学习率（缩放）
        print("lr_d", lr_d)  # 打印判别器学习率
        print("lr_g", lr_g)  # 打印生成器学习率
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))  # AE 优化器（编码/解码/量化相关参数）
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))  # 判别器优化器

        if self.scheduler_config is not None:  # 若配置了调度器
            scheduler = instantiate_from_config(self.scheduler_config)  # 实例化调度器配置对象

            print("Setting up LambdaLR scheduler...")  # 提示使用 LambdaLR
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),  # AE 的 LambdaLR 调度
                    'interval': 'step',  # 每步更新
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),  # 判别器的 LambdaLR 调度
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler  # 返回优化器与调度器
        return [opt_ae, opt_disc], []  # 未配置调度器则返回空列表

    def get_last_layer(self):  # 获取解码器最后一层权重（常用于损失归一化）
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):  # 记录可视化图像（重建与 EMA 重建）
        log = dict()  # 初始化日志字典
        x = self.get_input(batch, self.image_key)  # 获取输入图像
        x = x.to(self.device)  # 移动到当前设备
        if only_inputs:  # 若仅记录输入
            log["inputs"] = x  # 保存输入图像
            return log  # 返回日志
        xrec, _ = self(x)  # 重建图像
        if x.shape[1] > 3:  # 若通道数>3，通常为语义分割，需投影到 RGB 可视化
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)  # 输入转 RGB
            xrec = self.to_rgb(xrec)  # 重建转 RGB
        log["inputs"] = x  # 记录输入图像
        log["reconstructions"] = xrec  # 记录重建图像
        if plot_ema:  # 若需要绘制 EMA 版本
            with self.ema_scope():  # 切换 EMA 权重
                xrec_ema, _ = self(x)  # EMA 重建
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)  # 如为多通道则转 RGB
                log["reconstructions_ema"] = xrec_ema  # 记录 EMA 重建
        return log  # 返回图像日志

    def to_rgb(self, x):  # 将多通道语义分割张量随机投影到 RGB 以便可视化
        assert self.image_key == "segmentation"  # 仅当 image_key 为 segmentation 时使用
        if not hasattr(self, "colorize"):  # 若未注册 colorize buffer
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))  # 注册随机权重用于投影
        x = F.conv2d(x, weight=self.colorize)  # 使用 1x1 卷积进行线性投影到 RGB
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.  # 归一化到 [-1, 1]
        return x  # 返回 RGB 图像


class VQModelInterface(VQModel):  # 提供简化接口的 VQModel 子类（encode 返回预量化张量，decode 支持跳过量化）
    def __init__(self, embed_dim, *args, **kwargs):  # 初始化，强制传入 embed_dim
        super().__init__(embed_dim=embed_dim, *args, **kwargs)  # 调用父类构造
        self.embed_dim = embed_dim  # 保存嵌入维度

    def encode(self, x):  # 重载：返回预量化张量（不做码本匹配）
        h = self.encoder(x)  # 编码图像
        h = self.quant_conv(h)  # 通道映射到嵌入维度
        return h  # 返回预量化张量

    def decode(self, h, force_not_quantize=False):  # 解码接口，支持跳过量化
        # also go through quantization layer
        if not force_not_quantize:  # 默认仍通过量化器（更贴近真实流程）
            quant, emb_loss, info = self.quantize(h)  # 量化
        else:
            quant = h  # 强制跳过量化，直接使用输入
        quant = self.post_quant_conv(quant)  # 通道映射回 z_channels
        dec = self.decoder(quant)  # 解码为图像
        return dec  # 返回重建图像


class AutoencoderKL(pl.LightningModule):  # KL 正则的自编码器（变分自动编码器风格）
    def __init__(self,
                 ddconfig,  # 编解码器配置（要求 double_z=True 以输出均值与对数方差）
                 lossconfig,  # 损失配置（重建与对抗等）
                 embed_dim,  # 潜变量维度（VAE 的隐空间维度）
                 ckpt_path=None,  # 预训练权重路径
                 ignore_keys=[],  # 加载时忽略键前缀列表
                 image_key="image",  # batch 图像字段键名
                 colorize_nlabels=None,  # 语义分割随机投影标签数（可选）
                 monitor=None,  # 监控指标键名（可选）
                 ):
        super().__init__()  # LightningModule 初始化
        self.image_key = image_key  # 保存图像键名
        self.encoder = Encoder(**ddconfig)  # 实例化编码器
        self.decoder = Decoder(**ddconfig)  # 实例化解码器
        self.loss = instantiate_from_config(lossconfig)  # 从配置实例化损失模块
        assert ddconfig["double_z"]  # 断言编码器支持 double_z（输出均值和方差）
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)  # 将编码器输出的 (mu, logvar) 通道映射到目标维度
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)  # 解码前的通道映射
        self.embed_dim = embed_dim  # 保存潜变量维度
        if colorize_nlabels is not None:  # 若设置随机投影标签数
            assert type(colorize_nlabels)==int  # 必须为整数
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))  # 注册 buffer 用于可视化
        if monitor is not None:  # 若设置监控键
            self.monitor = monitor  # 保存监控键名
        if ckpt_path is not None:  # 若提供检查点路径
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)  # 加载权重

    def init_from_ckpt(self, path, ignore_keys=list()):  # 从检查点加载权重
        sd = torch.load(path, map_location="cpu")["state_dict"]  # 加载 state_dict
        keys = list(sd.keys())  # 键列表
        for k in keys:  # 遍历键
            for ik in ignore_keys:  # 遍历忽略前缀
                if k.startswith(ik):  # 匹配则删除该键
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)  # 非严格加载
        print(f"Restored from {path}")  # 打印恢复信息

    def encode(self, x):  # 编码得到后验分布（Diagonal Gaussian）
        h = self.encoder(x)  # 编码为特征图
        moments = self.quant_conv(h)  # 通过 1x1 卷积得到均值与对数方差（按通道拼接）
        posterior = DiagonalGaussianDistribution(moments)  # 构建对角高斯分布对象
        return posterior  # 返回后验分布

    def decode(self, z):  # 将潜变量解码为图像
        z = self.post_quant_conv(z)  # 通道映射到解码器输入通道
        dec = self.decoder(z)  # 解码为图像
        return dec  # 返回重建图像

    def forward(self, input, sample_posterior=True):  # 前向：编码-采样/取众数-解码
        posterior = self.encode(input)  # 后验分布
        if sample_posterior:  # 若采样（训练阶段）
            z = posterior.sample()  # 从对角高斯采样潜变量
        else:
            z = posterior.mode()  # 使用后验众数（推理阶段）
        dec = self.decode(z)  # 解码为图像
        return dec, posterior  # 返回重建与后验

    def get_input(self, batch, k):  # 取 batch 中图像并转为 NCHW
        x = batch[k]  # 取出图像
        if len(x.shape) == 3:  # 若缺少通道维
            x = x[..., None]  # 添加通道维度
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()  # 转为 NCHW 与 float
        return x  # 返回输入

    def training_step(self, batch, batch_idx, optimizer_idx):  # Lightning 训练步骤（多优化器）
        inputs = self.get_input(batch, self.image_key)  # 获取输入
        reconstructions, posterior = self(inputs)  # 前向：得到重建与后验分布

        if optimizer_idx == 0:  # 优化器 0：训练编码器+解码器+logvar（AE 路径）
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")  # 计算 AE 损失与日志
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 记录 AE 损失
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)  # 记录 AE 日志字典
            return aeloss  # 返回 AE 损失

        if optimizer_idx == 1:  # 优化器 1：训练判别器（对抗路径）
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")  # 判别器损失与日志

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 记录判别器损失
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)  # 记录日志
            return discloss  # 返回判别器损失

    def validation_step(self, batch, batch_idx):  # 验证步骤
        inputs = self.get_input(batch, self.image_key)  # 获取输入
        reconstructions, posterior = self(inputs)  # 前向
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")  # AE 损失与日志

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")  # 判别器损失与日志

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])  # 记录重建损失
        self.log_dict(log_dict_ae)  # 记录 AE 日志
        self.log_dict(log_dict_disc)  # 记录判别器日志
        return self.log_dict  # 返回日志字典

    def configure_optimizers(self):  # 配置优化器
        lr = self.learning_rate  # 学习率
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))  # AE 优化器
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))  # 判别器优化器
        return [opt_ae, opt_disc], []  # 返回优化器列表

    def get_last_layer(self):  # 获取解码器最后一层权重
        return self.decoder.conv_out.weight

    @torch.no_grad()  # 关闭梯度计算（推理／可视化）
    def log_images(self, batch, only_inputs=False, **kwargs):  # 记录输入、重建与采样图像
        log = dict()  # 初始化日志字典
        x = self.get_input(batch, self.image_key)  # 获取输入
        x = x.to(self.device)  # 移动到设备
        if not only_inputs:  # 若仅记录输入
            xrec, posterior = self(x)  # 重建与后验
            if x.shape[1] > 3:  # 多通道情况（如分割），进行随机投影到 RGB
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)  # 输入转 RGB
                xrec = self.to_rgb(xrec)  # 重建转 RGB
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))  # 采样随机潜变量并解码为样本
            log["reconstructions"] = xrec  # 记录重建
        log["inputs"] = x  # 始终记录输入
        return log  # 返回日志

    def to_rgb(self, x):  # 将分割张量随机投影到 RGB
        assert self.image_key == "segmentation"  # 仅用于分割任务
        if not hasattr(self, "colorize"):  # 若未注册投影权重
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))  # 注册随机权重
        x = F.conv2d(x, weight=self.colorize)  # 1x1 卷积实现线性投影
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.  # 归一化到 [-1, 1]
        return x  # 返回 RGB 图像


class IdentityFirstStage(torch.nn.Module):  # 恒等的一阶段模型（占位/直通接口）
    def __init__(self, *args, vq_interface=False, **kwargs):  # 可选：是否提供 VQ 接口兼容
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff  # 兼容旧逻辑
        super().__init__()  # 初始化父类

    def encode(self, x, *args, **kwargs):  # 直接返回输入（恒等变换）
        return x

    def decode(self, x, *args, **kwargs):  # 直接返回输入（恒等变换）
        return x

    def quantize(self, x, *args, **kwargs):  # 若需要与 VQ 接口兼容，则返回三元组
        if self.vq_interface:
            return x, None, [None, None, None]  # 返回占位的损失与信息
        return x  # 默认仅返回张量

    def forward(self, x, *args, **kwargs):  # 前向同样为恒等映射
        return x
