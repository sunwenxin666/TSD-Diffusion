"""SAMPLING ONLY."""  # 仅用于采样的模块

import torch  # 导入 PyTorch，用于张量计算
import numpy as np  # 导入 NumPy，用于数值计算
from tqdm import tqdm  # 导入进度条库 tqdm，用于显示采样进度
from functools import partial  # 导入 partial，用于构造偏函数

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like  # 从工具模块导入 DDIM 相关参数生成与噪声生成函数


class PLMSSampler(object):  # 定义 PLMS 采样器类（Pseudo Linear Multistep Sampler）
    def __init__(self, model, schedule="linear", **kwargs):  # 初始化采样器，传入扩散模型和时间步调度方式
        super().__init__()  # 调用父类初始化
        self.model = model  # 保存扩散模型实例
        self.ddpm_num_timesteps = model.num_timesteps  # 记录 DDPM 的总时间步数（通常为1000）
        self.schedule = schedule  # 保存时间步调度策略（如 linear）

    def register_buffer(self, name, attr):  # 将张量注册为对象属性，并确保在 CUDA 设备上
        if type(attr) == torch.Tensor:  # 如果属性是张量
            if attr.device != torch.device("cuda"):  # 如果不在 CUDA 设备上
                attr = attr.to(torch.device("cuda"))  # 移动到 CUDA
        setattr(self, name, attr)  # 将属性挂载到对象上

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True, sampled_timestep=None):  # 构建 PLMS 采样所需的时间步与参数
        if ddim_eta != 0:  # PLMS 要求 eta 为 0
            raise ValueError('ddim_eta must be 0 for PLMS')  # 抛出异常，提示必须为 0

        if sampled_timestep is None:  # 如果未指定采样的时间步集合
            self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                    num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)  # 计算 DDIM 时间步，按给定规则从 DDPM 时间步中均匀抽取
        else:
            self.ddim_timesteps = sampled_timestep  # 使用外部指定的采样时间步（如用于子采样或自定义步）

        alphas_cumprod = self.model.alphas_cumprod  # 从模型中获取累计 alpha（扩散过程中的衰减系数），长度为1000
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'  # 断言检查长度一致
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)  # 定义转换函数：拷贝、转为 float32，并移动到模型设备

        self.register_buffer('betas', to_torch(self.model.betas))  # 注册 betas（每步噪声强度），长度为1000
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))  # 注册累计 alpha
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev)) # 注册前一时刻的累计 alpha

        # calculations for diffusion q(x_t | x_{t-1}) and others  # 计算扩散过程 q(x_t | x_{t-1}) 相关的派生量
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))  # 累计 alpha 的平方根
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))  # 1 - 累计 alpha 的平方根
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))  # 1 - 累计 alpha 的对数
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))  # 累计 alpha 的倒数的平方根
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))  # (1/alpha - 1) 的平方根

        # ddim sampling parameters  # 计算 DDIM 采样所需参数（与 PLMS 结合使用）
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)  # 计算 DDIM 的 sigma、alpha 以及前一时刻 alpha
        self.register_buffer('ddim_sigmas', ddim_sigmas)  # 注册 DDIM 的噪声强度序列（通常长度为采样步数，如50）
        self.register_buffer('ddim_alphas', ddim_alphas)  # 注册 DDIM 的 alpha 序列
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)  # 注册 DDIM 的前一时刻 alpha 序列
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))  # 计算 DDIM 的 1 - alpha 的平方根
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))  # 根据 DDIM 推导得到与原始步数对应的 sigma
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)  # 注册上述 sigma

    @torch.no_grad()  # 关闭梯度计算，加速推理
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               sampled_timestep=None,
               prompt_mask=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...  # 注：条件需与模型匹配，如文本编码向量
               **kwargs
               ):
        if conditioning is not None:  # 如果提供了条件（如文本编码或引导条件）
            if isinstance(conditioning, dict):  # 如果条件是字典（多路条件）
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]  # 取第一个条件的 batch 大小
                if cbs != batch_size:  # 检查与目标 batch 是否一致
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")  # 发出警告
            else:
                if conditioning.shape[0] != batch_size:  # 条件为张量时检查维度一致性
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")  # 发出警告

        if sampled_timestep is not None:  # 如果指定了自定义时间步集合
            sampled_timestep = sorted(sampled_timestep)  # 按升序排序，确保时间顺序
                
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose, sampled_timestep=sampled_timestep)  # 修改（生成）采样时间步与参数
        # sampling  # 开始采样
        C, H, W = shape  # 通道数与图像尺寸
        size = (batch_size, C, H, W)  # 最终张量尺寸
        print(f'Data shape for PLMS sampling is {size}')   # 打印采样张量形状
        samples, intermediates = self.plms_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    prompt_mask=prompt_mask,
                                                    )  # 调用 PLMS 采样过程
        return samples, intermediates  # 返回最终样本与中间过程

    @torch.no_grad()  # 关闭梯度计算
    def plms_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, 
                      prompt_mask=None, ):  # 核心采样逻辑（PLMS 方法）
        device = self.model.betas.device  # 获取模型所在设备
        b = shape[0]  # batch 大小
        if x_T is None:  # 如果未提供初始噪声图像
            img = torch.randn(shape, device=device)  # 使用正态分布初始化噪声
        else:
            img = x_T  # 使用外部提供的初始噪声

        if timesteps is None:  # 如果未指定时间步序列
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps  # 使用 DDPM 或 DDIM 时间步
        elif timesteps is not None and not ddim_use_original_steps:  # 如果指定了时间步且不使用原始 DDPM 步数
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1  # 计算子集结束位置
            timesteps = self.ddim_timesteps[:subset_end]  # 取子集时间步

        timesteps = np.asarray(timesteps)  # 转为 NumPy 数组便于反转与索引
        intermediates = {'x_inter': [img], 'pred_x0': [img]}  # 存储中间图像和预测的 x0
        time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)  # 构造反向时间序列（从大到小）
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]  # 总步数
        print(f"Running PLMS Sampling with {total_steps} timesteps")  # 打印总步数

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)  # 使用 tqdm 构造进度条
        old_eps = []  # 保存历史噪声预测 e_t，用于多步法（Adams-Bashforth）

        for i, step in enumerate(iterator):  # 遍历每个时间步
            index = total_steps - i - 1  # 根据进度条位置计算索引（与时间步一致）
            ts = torch.full((b,), step, device=device, dtype=torch.long)  # 构造当前步的时间张量
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)  # 下一个时间步

            if mask is not None:  # 如果有掩码（如用于图像修复）
                assert x0 is not None  # 必须提供原始图像 x0
                img_orig = self.model.q_sample(x0, ts)  # 对 x0 添加噪声得到当前步的参考图像（近似前向过程）
                img = img_orig * mask + (1. - mask) * img  # 使用掩码混合：mask=1 用原图，mask=0 用当前生成
            
            if prompt_mask is not None and prompt_mask[i] == 0:  # 如果提供提示掩码并指示该步使用无条件生成
                outs = self.p_sample_plms(img, unconditional_conditioning, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=1.,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next)  # 使用无条件引导生成该一步
            else:  # 否则使用条件引导（如文本条件）
                outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        old_eps=old_eps, t_next=ts_next)  # 常规条件生成
                
            img, pred_x0, e_t = outs  # 更新当前图像、预测的 x0 以及当前步的噪声预测
            old_eps.append(e_t)  # 记录历史噪声预测
            if len(old_eps) >= 4:  # 保留最近 3 个历史 e_t（最多 4 个，随后丢弃最旧的一个）
                old_eps.pop(0)
            if callback: callback(i)  # 若提供回调函数则调用（如自定义日志）
            if img_callback: img_callback(pred_x0, i)  # 若提供图像回调则返回当前预测的 x0

            if index % log_every_t == 0 or index == total_steps - 1:  # 控制日志频率或最后一步
                intermediates['x_inter'].append(img)  # 保存中间生成图像
                intermediates['pred_x0'].append(pred_x0)  # 保存中间预测的 x0

        return img, intermediates  # 返回最终图像与中间结果

    @torch.no_grad()  # 关闭梯度计算
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):  # 单步 PLMS 采样（包含多步法外推）
        b, *_, device = *x.shape, x.device  # 解析 batch 大小与设备

        def get_model_output(x, t):  # 内部函数：根据是否使用引导，计算模型在 (x, t) 上的噪声预测 e_t
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:  # 无条件或无引导时
                e_t = self.model.apply_model(x, t, c)  # 直接调用模型得到噪声预测
            else:  # 使用 classifier-free guidance（无分类器引导的条件引导）
                x_in = torch.cat([x] * 2)  # 将输入 x 复制拼接（用于条件与无条件）
                t_in = torch.cat([t] * 2)  # 时间步同样复制
                c_in = torch.cat([unconditional_conditioning, c])  # 条件拼接：前半无条件，后半有条件
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)  # 模型输出拆分为无条件与有条件部分
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)  # 按比例进行引导插值

            if score_corrector is not None:  # 如果提供了分数修正器（如用于修正噪声预测）
                assert self.model.parameterization == "eps"  # 要求模型参数化为噪声预测 eps
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)  # 调用修正器修改 e_t

            return e_t  # 返回噪声预测

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas  # 根据是否使用原始步数选择 alpha 序列
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev  # 选择前一时刻 alpha
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas  # 选择 sqrt(1-alpha)
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas  # 选择噪声强度 sigma

        def get_x_prev_and_pred_x0(e_t, index):  # 根据当前噪声预测 e_t，计算上一时刻的图像 x_{t-1} 与预测的 x0
            # select parameters corresponding to the currently considered timestep  # 为当前步索引选择参数
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)  # 当前步的 alpha
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)  # 上一步的 alpha
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)  # 当前步的噪声强度
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)  # 当前步的 sqrt(1-alpha)

            # current prediction for x_0  # 根据公式预测 x0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()  # 由 x_t 和 e_t 反推 x0
            if quantize_denoised:  # 若需要对预测的 x0 进行量化（适用于 VQ 模型）
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)  # 量化 x0
            # direction pointing to x_t  # 计算指向 x_t 的方向项（DDIM/PLMS 推导公式）
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t  # 方向项
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature  # 采样噪声，按温度缩放
            if noise_dropout > 0.:  # 如果设置噪声 dropout 进行随机置零
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)  # 对噪声进行 dropout
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise  # 组合得到上一时刻图像 x_{t-1}
            return x_prev, pred_x0  # 返回上一时刻图像与预测的 x0

        e_t = get_model_output(x, t)  # 计算当前步的噪声预测
        if len(old_eps) == 0:  # 如果没有历史噪声（第一步）
            # Pseudo Improved Euler (2nd order)  # 伪改进欧拉法（二阶），先预测一步再做校正
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)  # 使用当前噪声预测得到 x_{t-1}
            e_t_next = get_model_output(x_prev, t_next)  # 在预测的下一位置计算噪声
            e_t_prime = (e_t + e_t_next) / 2  # 二阶欧拉平均，得到改进后的噪声预测
        elif len(old_eps) == 1:  # 若仅有一个历史噪声
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)  # 二阶 Adams-Bashforth 伪线性多步法
            e_t_prime = (3 * e_t - old_eps[-1]) / 2  # 使用 2 步线性组合得到外推噪声
        elif len(old_eps) == 2:  # 若有两个历史噪声
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)  # 三阶 Adams-Bashforth 伪线性多步法
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12  # 三步线性组合
        elif len(old_eps) >= 3:  # 若有三个及以上历史噪声
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)  # 四阶 Adams-Bashforth 伪线性多步法
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24  # 四步线性组合

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)  # 使用外推噪声计算上一时刻图像与预测 x0

        return x_prev, pred_x0, e_t  # 返回上一时刻图像、预测 x0 和当前噪声
