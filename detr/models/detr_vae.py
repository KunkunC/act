# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
这是一个基于 DETR (DEtection TRansformer) 的 VAE (变分自编码器) 模型文件。
用于机器人模仿学习 (ACT: Action Chunking with Transformers)，
核心思路：用 CVAE (条件变分自编码器) 将动作序列编码为隐变量 z，再通过 Transformer 解码器解码出动作序列。
"""
import torch                      # PyTorch 深度学习框架
from torch import nn               # nn 模块包含所有神经网络层的定义（Linear, Conv2d, Embedding 等）
from torch.autograd import Variable # Variable 是 Tensor 的包装，用于自动求导（旧版 API，现在 Tensor 本身就支持）
from .backbone import build_backbone  # 从同包导入 backbone 构建函数（通常是 ResNet 等卷积网络，用于提取图像特征）
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
# build_transformer: 构建完整的 Transformer（含 encoder + decoder）
# TransformerEncoder: Transformer 编码器，由多层 TransformerEncoderLayer 堆叠而成
# TransformerEncoderLayer: 单层 Transformer 编码器，包含自注意力 + 前馈网络

import numpy as np                 # NumPy 数值计算库

import IPython
e = IPython.embed                  # 调试用：在代码中插入 e() 可进入交互式 IPython 终端


def reparametrize(mu, logvar):
    """
    重参数化技巧 (Reparameterization Trick)。
    VAE 的核心技巧：将从 N(mu, sigma^2) 采样转化为 mu + sigma * epsilon，
    其中 epsilon ~ N(0, 1)，这样梯度可以通过 mu 和 logvar 反向传播。

    参数:
        mu:     均值向量，形状 (batch_size, latent_dim)
        logvar: 对数方差向量，形状 (batch_size, latent_dim)
    返回:
        从 N(mu, sigma^2) 中采样的隐变量 z
    """
    std = logvar.div(2).exp()
    # logvar / 2 = log(sigma)，再 exp() 得到标准差 sigma
    # 数学推导: exp(log(var) / 2) = exp(log(sigma^2) / 2) = exp(log(sigma)) = sigma

    eps = Variable(std.data.new(std.size()).normal_())
    # 从标准正态分布 N(0,1) 采样一个与 std 同形状的噪声 epsilon
    # std.data.new() 创建与 std 相同设备(CPU/GPU)和类型的新张量
    # .normal_() 原地填充标准正态随机数

    return mu + std * eps
    # 重参数化公式: z = mu + sigma * epsilon
    # 等价于从 N(mu, sigma^2) 中采样，但允许梯度反传


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    生成正弦位置编码表 (Sinusoidal Positional Encoding)。
    来自 "Attention is All You Need" 论文，用固定的正弦/余弦函数为序列中的每个位置生成唯一编码。
    不需要学习，通过不同频率的正弦波组合来表示位置信息。

    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  -- 偶数维度用 sin
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))  -- 奇数维度用 cos

    参数:
        n_position: 位置总数（序列最大长度）
        d_hid:      隐藏层维度（每个位置编码的维度）
    返回:
        形状为 (1, n_position, d_hid) 的位置编码张量
    """
    def get_position_angle_vec(position):
        # 对某个位置 position，计算它在每个维度上的角度值
        # position / 10000^(2*(hid_j//2)/d_hid)
        # hid_j//2 使得相邻的两个维度（sin 和 cos）使用相同的频率
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # 生成 (n_position, d_hid) 的角度矩阵，每一行是一个位置的所有维度的角度值

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度 (0,2,4,...) 取 sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度 (1,3,5,...) 取 cos

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    # 转为 PyTorch 张量并增加 batch 维度，形状: (1, n_position, d_hid)
    # unsqueeze(0) 在第0维插入一个维度，方便后续 batch 广播


class DETRVAE(nn.Module):
    """
    DETR-VAE 模型：结合了 DETR (DEtection TRansformer) 和 CVAE (条件变分自编码器) 的架构。

    整体流程:
    1. [训练时] Encoder 部分: 将动作序列 + 机器人状态编码为隐变量 z (通过 VAE 的 encoder)
    2. [推理时] 隐变量 z 直接从标准正态分布 N(0,I) 采样
    3. Decoder 部分: 将图像特征 + 机器人状态 + 隐变量 z 输入 Transformer decoder，解码出动作序列

    这个架构的核心思想是 ACT (Action Chunking with Transformers):
    - 一次性预测未来多步动作（chunking），而非逐步预测
    - 用 CVAE 建模动作序列的多模态分布（同一观测可能对应多种合理动作）
    """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """
        初始化 DETRVAE 模型。

        参数:
            backbones:     图像特征提取骨干网络列表（通常是 ResNet），用于将原始图像转换为特征图
            transformer:   Transformer 模块（包含 encoder 和 decoder），是动作解码的核心
            encoder:       CVAE 的编码器（独立的 Transformer Encoder），用于训练时将动作序列编码为隐变量
            state_dim:     机器人状态/动作的维度（例如 14 = 7个关节位置 * 2只手臂）
            num_queries:   查询数量，即模型一次预测的动作步数（如 100 步）
            camera_names:  相机名称列表（如 ['top', 'left_wrist', 'right_wrist']），支持多视角输入
        """
        super().__init__()
        # 调用 nn.Module 的构造函数，这是 PyTorch 定义网络的标准写法，必须调用

        self.num_queries = num_queries
        # 保存查询数量，即模型一次预测多少步动作（action chunk size）

        self.camera_names = camera_names
        # 保存相机名称列表，forward 中遍历每个相机提取图像特征

        self.transformer = transformer
        # 保存 Transformer 模块（DETR 的核心，包含 encoder + decoder）
        # encoder 处理图像特征，decoder 通过 query 解码出动作序列

        self.encoder = encoder
        # 保存 CVAE 的 Transformer Encoder（注意：这和 transformer 里的 encoder 不是同一个）
        # 这个 encoder 专门用于训练时将动作序列压缩为隐变量 z

        hidden_dim = transformer.d_model
        # 获取 Transformer 的隐藏层维度（如 256 或 512），所有子模块的维度要与之对齐

        self.action_head = nn.Linear(hidden_dim, state_dim)
        # 动作预测头：全连接层，将 Transformer 的输出 (hidden_dim) 映射到动作空间 (state_dim)
        # nn.Linear(in_features, out_features): 线性变换 y = xW^T + b
        # 例如: 256 -> 14，输出每一步的关节角度/位置
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        # 填充预测头：预测每个时间步是否为 padding（无效动作）
        # 输出 1 维 logit，用 sigmoid 可得到是否为 padding 的概率
        # 因为 action chunk 可能不需要填满所有 num_queries 步

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 查询嵌入 (Query Embedding)：DETR 的核心概念
        # nn.Embedding(num_embeddings, embedding_dim): 可学习的查找表
        # 每个 query 是一个可学习的向量，代表一个"检测槽位"
        # 在 ACT 中，每个 query 对应预测一个时间步的动作
        # 形状: (num_queries, hidden_dim)，如 (100, 256)

        if backbones is not None:
            # 有图像输入的情况（正常模式：从图像观测预测动作）

            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            # 输入投影层：1x1 卷积，将 backbone 输出的通道数调整为 hidden_dim
            # nn.Conv2d(in_channels, out_channels, kernel_size): 2D 卷积
            # kernel_size=1 的卷积等价于逐像素的线性变换，只改变通道数不改变空间尺寸
            # 例如: ResNet18 输出 512 通道 -> 投影为 256 通道

            self.backbones = nn.ModuleList(backbones)
            # nn.ModuleList: 将多个子模块存储为列表
            # 与普通 Python list 不同，ModuleList 会正确注册子模块的参数
            # 这样 model.parameters() 能遍历到 backbone 的参数，优化器才能更新它们

            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            # 机器人状态投影：将 14 维的关节状态 (qpos) 投影到 hidden_dim
            # 14 维通常 = 7个关节角度 * 2只手臂

        else:
            # 无图像输入的情况（纯状态模式：仅从机器人状态 + 环境状态预测动作）
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            # 机器人状态投影，同上

            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            # 环境状态投影：将 7 维的环境状态投影到 hidden_dim

            self.pos = torch.nn.Embedding(2, hidden_dim)
            # 位置嵌入：2 个可学习的位置编码（对应 robot_state 和 env_state 两个 token）

            self.backbones = None
            # 标记无 backbone

        # ============ CVAE Encoder 额外参数 ============
        # 以下参数仅在训练时使用，用于 VAE 的编码过程

        self.latent_dim = 32 # final size of latent z # TODO tune
        # 隐变量 z 的维度（VAE 的瓶颈维度）
        # 信息被压缩到 32 维的隐空间中，控制模型的表达能力和正则化强度

        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        # CLS token 嵌入：类似 BERT 的 [CLS] token
        # 只有 1 个嵌入向量，形状 (1, hidden_dim)
        # 放在序列开头，其输出用于聚合整个序列的信息，最终映射到隐变量 z

        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        # 动作投影层：将 14 维的动作向量投影到 hidden_dim
        # 用于 CVAE encoder 的输入——将动作序列的每一步投影为嵌入向量

        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        # 关节状态投影层：将 14 维的关节位置 (qpos) 投影到 hidden_dim
        # 用于 CVAE encoder 的输入——让 encoder 同时看到当前状态和动作序列

        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        # 隐变量投影层：将 encoder 的 CLS 输出映射为 mu 和 logvar
        # 输出维度是 latent_dim*2 = 64，前 32 维是均值 mu，后 32 维是对数方差 logvar
        # 这两个参数定义了隐变量 z 的高斯分布 q(z|x, a)

        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq
        # 注册正弦位置编码表为 buffer
        # register_buffer: 注册为模型的一部分（会随模型保存/加载/移动到GPU），但不参与梯度更新
        # 位置数量 = 1(CLS) + 1(qpos) + num_queries(动作序列)
        # 为 CVAE encoder 的输入序列提供位置信息

        # ============ CVAE Decoder 额外参数 ============

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        # 隐变量输出投影：将采样得到的 32 维隐变量 z 映射回 hidden_dim
        # 映射后的向量将作为额外输入送入 Transformer decoder

        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        # 额外的可学习位置嵌入，包含 2 个向量:
        # 第 0 个: 用于机器人本体感觉 (proprioception/qpos) token 的位置编码
        # 第 1 个: 用于隐变量 (latent) token 的位置编码
        # 这些位置编码告诉 Transformer 这两个额外 token 分别是什么

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        DETRVAE 的前向传播。

        整体流程:
        1. [仅训练] 通过 CVAE encoder 将 (qpos, actions) 编码为隐变量 z
        2. [仅推理] 从标准正态 N(0,I) 采样隐变量 z
        3. 通过 backbone 提取图像特征
        4. 将图像特征 + qpos + z 送入 Transformer decoder 解码出动作序列

        参数:
            qpos:      机器人关节位置，形状 (batch_size, 14)
            image:     多相机图像，形状 (batch_size, num_cam, 3, height, width)
            env_state: 环境状态（本模型中通常为 None）
            actions:   真实动作序列，形状 (batch_size, seq_len, 14)，仅训练时提供
            is_pad:    padding 掩码，形状 (batch_size, seq_len)，True 表示该步是填充的

        返回:
            a_hat:      预测的动作序列 (batch_size, num_queries, state_dim)
            is_pad_hat: 预测的 padding 标记 (batch_size, num_queries, 1)
            [mu, logvar]: VAE 的均值和对数方差（训练时非 None，推理时为 None）
        """
        is_training = actions is not None # train or val
        # 判断是训练还是推理：训练时会传入真实动作序列 actions

        bs, _ = qpos.shape
        # 获取 batch_size; qpos 形状是 (batch_size, 14)，_ 丢弃第二个维度

        ### ============ 第一步：获取隐变量 z ============
        if is_training:
            # 训练模式：通过 CVAE Encoder 从动作序列中提取隐变量

            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            # 将动作序列投影到嵌入空间
            # actions: (bs, seq, 14) -> action_embed: (bs, seq, hidden_dim)

            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            # 将关节位置投影到嵌入空间
            # qpos: (bs, 14) -> qpos_embed: (bs, hidden_dim)

            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            # 在第 1 维插入一个维度，使其可以和 action_embed 在序列维度拼接
            # (bs, hidden_dim) -> (bs, 1, hidden_dim)
            # unsqueeze: 增加一个大小为 1 的维度

            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            # 取出 CLS token 的嵌入向量
            # nn.Embedding.weight 就是嵌入矩阵本身

            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            # 先增加 batch 维: (1, hidden_dim) -> (1, 1, hidden_dim)
            # 再沿 batch 维复制 bs 次: (1, 1, hidden_dim) -> (bs, 1, hidden_dim)
            # repeat(bs, 1, 1): 第0维重复bs次，第1、2维不重复

            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+2, hidden_dim)
            # 沿序列维度拼接: [CLS] + [qpos] + [action_1, action_2, ..., action_T]
            # torch.cat: 沿指定维度拼接多个张量

            encoder_input = encoder_input.permute(1, 0, 2) # (seq+2, bs, hidden_dim)
            # 调整维度顺序: (bs, seq+2, hidden_dim) -> (seq+2, bs, hidden_dim)
            # permute: 重新排列维度的顺序
            # PyTorch 的 Transformer 默认输入格式是 (seq_len, batch, feature)

            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            # 为 CLS 和 qpos 创建 padding 掩码，都设为 False（不是 padding）
            # torch.full(size, fill_value): 创建指定大小的张量并填充指定值
            # .to(device): 将张量移动到与 qpos 相同的设备（CPU 或 GPU）

            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+2)
            # 拼接掩码: [False, False, is_pad_action_1, ..., is_pad_action_T]
            # CLS 和 qpos 永远不被 mask，动作序列中超出实际长度的部分被 mask

            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            # 获取正弦位置编码的副本
            # .clone(): 深拷贝张量
            # .detach(): 从计算图中分离（位置编码不需要梯度）
            # 形状: (1, seq+2, hidden_dim)

            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
            # 调整维度: (1, seq+2, hidden_dim) -> (seq+2, 1, hidden_dim)
            # 与 encoder_input 的维度对齐，第 1 维会自动广播到 bs

            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            # 将拼接好的序列送入 CVAE 的 Transformer Encoder
            # encoder_input: 输入序列 (seq+2, bs, hidden_dim)
            # pos: 位置编码，加到自注意力的 Q 和 K 上
            # src_key_padding_mask: 告诉注意力机制哪些位置是 padding，应该被忽略
            # 输出形状: (seq+2, bs, hidden_dim)

            encoder_output = encoder_output[0] # take cls output only
            # 只取第 0 个位置的输出，即 CLS token 的输出
            # CLS token 通过自注意力聚合了整个序列的信息
            # 形状: (bs, hidden_dim)

            latent_info = self.latent_proj(encoder_output)
            # 将 CLS 输出投影为隐变量的参数 (mu 和 logvar)
            # (bs, hidden_dim) -> (bs, latent_dim*2) = (bs, 64)

            mu = latent_info[:, :self.latent_dim]
            # 前 32 维是均值 mu，形状: (bs, 32)
            # 切片操作: [:, :32] 取所有 batch 的前 32 个特征

            logvar = latent_info[:, self.latent_dim:]
            # 后 32 维是对数方差 logvar，形状: (bs, 32)
            # 切片操作: [:, 32:] 取所有 batch 的后 32 个特征

            latent_sample = reparametrize(mu, logvar)
            # 用重参数化技巧从 N(mu, sigma^2) 采样隐变量 z
            # 形状: (bs, 32)

            latent_input = self.latent_out_proj(latent_sample)
            # 将 32 维的隐变量投影回 hidden_dim，准备送入 Transformer decoder
            # (bs, 32) -> (bs, hidden_dim)

        else:
            # 推理模式：没有真实动作序列，直接从先验分布 N(0, I) 采样
            mu = logvar = None
            # 推理时不需要 mu 和 logvar

            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            # 创建全零的隐变量（等价于取先验分布的均值）
            # 形状: (bs, 32)
            # 推理时用零向量而非随机采样，以获得确定性的输出

            latent_input = self.latent_out_proj(latent_sample)
            # 同样投影回 hidden_dim
            # (bs, 32) -> (bs, hidden_dim)

        ### ============ 第二步：提取观测特征并解码动作 ============
        if self.backbones is not None:
            # 有图像输入的模式（主要模式）

            # Image observation features and position embeddings
            all_cam_features = []   # 存放所有相机的特征图
            all_cam_pos = []        # 存放所有相机特征图的位置编码
            for cam_id, cam_name in enumerate(self.camera_names):
                # 遍历每个相机

                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                # 用 backbone（如 ResNet）提取第 cam_id 个相机的图像特征
                # image[:, cam_id]: 取出当前相机的图像 (bs, 3, H, W)
                # features: 多尺度特征图列表
                # pos: 对应的位置编码列表
                # 注意: 所有相机共享同一个 backbone (backbones[0])，这是硬编码的

                features = features[0] # take the last layer feature
                # 取最后一层（最深层）的特征图
                # 形状: (bs, backbone_channels, h, w)，如 (bs, 512, 15, 20)

                pos = pos[0]
                # 取对应的位置编码
                # 形状: (bs, hidden_dim, h, w)

                all_cam_features.append(self.input_proj(features))
                # 用 1x1 卷积将特征图通道数从 backbone_channels 调整为 hidden_dim
                # (bs, 512, h, w) -> (bs, hidden_dim, h, w)

                all_cam_pos.append(pos)
                # 位置编码直接加入列表

            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # 将机器人关节状态投影到 hidden_dim
            # (bs, 14) -> (bs, hidden_dim)
            # proprio = proprioception（本体感觉），即机器人自身的状态

            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            # 沿宽度维度 (axis=3) 拼接所有相机的特征图
            # 例如 3 个相机: (bs, hidden_dim, h, w) x 3 -> (bs, hidden_dim, h, 3*w)
            # 这样把多相机特征"横向拼接"成一个大特征图

            pos = torch.cat(all_cam_pos, axis=3)
            # 同样拼接位置编码
            # (bs, hidden_dim, h, w) x 3 -> (bs, hidden_dim, h, 3*w)

            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
            # 将所有信息送入 Transformer 进行解码
            # src:                      图像特征 (bs, hidden_dim, h, 3*w)
            # None:                     mask（未使用）
            # self.query_embed.weight:  查询嵌入 (num_queries, hidden_dim)，每个 query 解码一步动作
            # pos:                      位置编码 (bs, hidden_dim, h, 3*w)
            # latent_input:             隐变量 (bs, hidden_dim)
            # proprio_input:            本体感觉 (bs, hidden_dim)
            # additional_pos_embed:     额外位置嵌入 (2, hidden_dim)
            # [0] 取 decoder 的最后一层输出
            # hs 形状: (bs, num_queries, hidden_dim)

        else:
            # 无图像输入的模式（纯状态模式，用于简单环境）

            qpos = self.input_proj_robot_state(qpos)
            # 投影机器人状态: (bs, 14) -> (bs, hidden_dim)

            env_state = self.input_proj_env_state(env_state)
            # 投影环境状态: (bs, 7) -> (bs, hidden_dim)

            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            # 拼接为序列: (bs, 2, hidden_dim)

            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
            # 送入 Transformer 解码
            # self.pos.weight: 2 个可学习位置编码
            # 返回: (bs, num_queries, hidden_dim)

        a_hat = self.action_head(hs)
        # 动作预测: 通过线性层将 Transformer 输出映射为动作
        # (bs, num_queries, hidden_dim) -> (bs, num_queries, state_dim)
        # 即预测 num_queries 步动作，每步 state_dim=14 维

        is_pad_hat = self.is_pad_head(hs)
        # padding 预测: 预测每步是否为填充
        # (bs, num_queries, hidden_dim) -> (bs, num_queries, 1)

        return a_hat, is_pad_hat, [mu, logvar]
        # 返回: 预测动作序列、padding 预测、VAE 参数
        # mu 和 logvar 用于计算 KL 散度损失（训练时）



class CNNMLP(nn.Module):
    """
    CNN + MLP 基线模型：简单的卷积网络 + 多层感知机。
    作为对比基线，不使用 Transformer 和 VAE，直接从图像特征 + 状态回归动作。
    结构简单但表达能力有限，无法建模多模态动作分布。
    """
    def __init__(self, backbones, state_dim, camera_names):
        """
        初始化 CNN-MLP 模型。

        参数:
            backbones:     每个相机对应一个 backbone（注意：与 DETRVAE 不同，这里每个相机有独立 backbone）
            state_dim:     机器人状态/动作的维度
            camera_names:  相机名称列表
        """
        super().__init__()
        # 调用父类 nn.Module 的构造函数

        self.camera_names = camera_names
        # 保存相机名称列表

        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        # 动作预测头: 1000 -> state_dim
        # 注意: 这个层在当前代码中实际未被使用（forward 中用的是 self.mlp）

        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            # 用 ModuleList 注册所有 backbone
            # 与 DETRVAE 不同，每个相机有自己的 backbone

            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    # 第一个卷积: 降低通道数到 128，5x5 卷积核
                    nn.Conv2d(128, 64, kernel_size=5),
                    # 第二个卷积: 继续降低通道数到 64
                    nn.Conv2d(64, 32, kernel_size=5)
                    # 第三个卷积: 最终降到 32 通道
                    # nn.Sequential: 将多个层按顺序组合，forward 时依次执行
                    # 注意: 这里没有激活函数和池化层，是一个非常简化的设计
                )
                backbone_down_projs.append(down_proj)

            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)
            # 注册所有降维卷积模块

            mlp_in_dim = 768 * len(backbones) + 14
            # MLP 输入维度 = 每个相机展平后的特征维度 * 相机数 + 机器人状态维度
            # 768 是每个相机特征展平后的大小（32通道 * 某空间尺寸）
            # 14 是机器人关节状态维度

            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
            # 创建 MLP（多层感知机）
            # 结构: mlp_in_dim -> 1024 -> ReLU -> 1024 -> ReLU -> 14
            # 直接从拼接的特征回归 14 维动作

        else:
            raise NotImplementedError
            # 必须提供 backbone，否则抛出未实现异常

    def forward(self, qpos, image, env_state, actions=None):
        """
        CNN-MLP 的前向传播。

        参数:
            qpos:      机器人关节位置 (batch_size, 14)
            image:     多相机图像 (batch_size, num_cam, 3, H, W)
            env_state: 环境状态（未使用）
            actions:   真实动作（未使用，仅用于接口兼容）
        返回:
            a_hat: 预测的动作 (batch_size, 14)，注意只预测单步动作
        """
        is_training = actions is not None # train or val
        # 判断训练/推理模式（此模型中该标记实际未使用）

        bs, _ = qpos.shape
        # 获取 batch_size

        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            # 遍历每个相机

            features, pos = self.backbones[cam_id](image[:, cam_id])
            # 用对应的 backbone 提取特征
            # 注意: 每个相机用不同的 backbone（backbones[cam_id]）

            features = features[0] # take the last layer feature
            # 取最后一层特征图

            pos = pos[0] # not used
            # 位置编码（此模型中未使用）

            all_cam_features.append(self.backbone_down_projs[cam_id](features))
            # 通过降维卷积减少特征图的通道数和空间尺寸
            # (bs, backbone_channels, h, w) -> (bs, 32, h', w')

        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
            # 将每个相机的特征图展平为一维向量
            # (bs, 32, h', w') -> (bs, 32*h'*w') = (bs, 768)
            # reshape([bs, -1]): 保持 batch 维不变，其余维度展平，-1 表示自动计算

        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        # 拼接所有相机的展平特征
        # (bs, 768) x num_cam -> (bs, 768*num_cam)

        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        # 拼接图像特征和机器人状态
        # (bs, 768*num_cam + 14)

        a_hat = self.mlp(features)
        # 通过 MLP 预测动作
        # (bs, 768*num_cam+14) -> (bs, 14)

        return a_hat
        # 返回预测的单步动作


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    """
    构建一个多层感知机 (MLP / 全连接神经网络)。

    参数:
        input_dim:    输入维度
        hidden_dim:   隐藏层维度
        output_dim:   输出维度
        hidden_depth: 隐藏层层数（0 表示直连，无隐藏层）
    返回:
        nn.Sequential: 顺序容器，包含线性层和激活函数

    示例:
        hidden_depth=0: input -> output (单层线性)
        hidden_depth=1: input -> hidden -> ReLU -> output (一层隐藏)
        hidden_depth=2: input -> hidden -> ReLU -> hidden -> ReLU -> output (两层隐藏)
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
        # 无隐藏层: 直接 input_dim -> output_dim

    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        # 第一层: input_dim -> hidden_dim + ReLU 激活
        # nn.ReLU(inplace=True): 原地操作节省内存，ReLU(x) = max(0, x)

        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            # 中间层: hidden_dim -> hidden_dim + ReLU
            # 循环 (hidden_depth - 1) 次

        mods.append(nn.Linear(hidden_dim, output_dim))
        # 最后一层: hidden_dim -> output_dim（无激活函数，直接输出）

    trunk = nn.Sequential(*mods)
    # nn.Sequential: 将所有层按顺序封装为一个模块
    # *mods: 解包列表作为位置参数
    # forward 时按顺序执行: x -> layer1 -> layer2 -> ... -> output

    return trunk


def build_encoder(args):
    """
    构建 CVAE 的 Transformer Encoder。
    这是独立于 DETR 主 Transformer 的编码器，专门用于将动作序列编码为隐变量 z。

    参数:
        args: 包含模型超参数的命名空间对象
    返回:
        TransformerEncoder: 由多层 TransformerEncoderLayer 组成的编码器
    """
    d_model = args.hidden_dim             # 模型隐藏维度，如 256
    dropout = args.dropout                # Dropout 比率，如 0.1，用于防止过拟合
    nhead = args.nheads                   # 多头注意力的头数，如 8
    dim_feedforward = args.dim_feedforward # 前馈网络的中间维度，如 2048
    num_encoder_layers = args.enc_layers  # 编码器层数，如 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm      # 是否使用 Pre-LayerNorm（先归一化再做注意力）
    activation = "relu"                   # 前馈网络中的激活函数

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    # 创建单层 Transformer 编码器
    # 包含: 多头自注意力 -> Add&Norm -> 前馈网络 -> Add&Norm

    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    # 如果使用 Pre-Norm，在最后加一个 LayerNorm
    # nn.LayerNorm: 层归一化，沿特征维度进行归一化

    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    # 创建完整编码器: 将 encoder_layer 复制 num_encoder_layers 次并堆叠
    # 每一层的参数是独立的（不共享权重）

    return encoder


def build(args):
    """
    构建完整的 DETRVAE 模型（工厂函数）。
    将各个组件（backbone、transformer、encoder）组装在一起。

    参数:
        args: 包含所有模型超参数的命名空间对象
    返回:
        DETRVAE: 完整的模型实例
    """
    state_dim = 14 # TODO hardcode
    # 机器人状态/动作维度，硬编码为 14（7关节 * 2手臂）

    backbones = []
    backbone = build_backbone(args)
    # 构建图像特征提取 backbone（如 ResNet18）
    backbones.append(backbone)
    # 加入列表（只用一个 backbone，所有相机共享）

    transformer = build_transformer(args)
    # 构建 DETR 的 Transformer（包含 encoder 和 decoder）

    encoder = build_encoder(args)
    # 构建 CVAE 的独立 Transformer Encoder

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )
    # 实例化 DETRVAE 模型，传入所有组件

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 统计模型的可训练参数总数
    # p.numel(): 返回张量中元素的总数
    # p.requires_grad: 过滤掉冻结的参数

    print("number of parameters: %.2fM" % (n_parameters/1e6,))
    # 打印参数量（以百万为单位）

    return model

def build_cnnmlp(args):
    """
    构建 CNN-MLP 基线模型（工厂函数）。

    参数:
        args: 包含模型超参数的命名空间对象
    返回:
        CNNMLP: 基线模型实例
    """
    state_dim = 14 # TODO hardcode
    # 动作维度，硬编码为 14

    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
    # 为每个相机创建独立的 backbone（与 DETRVAE 不同）
    # _: 下划线表示不需要循环变量的值

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )
    # 实例化 CNN-MLP 模型

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))
    # 打印可训练参数量

    return model
