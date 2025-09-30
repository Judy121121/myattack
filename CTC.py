import torch
import numpy as np


def sequence_cw_loss(sequence_logits, targets, kappa=-0., tar=False):
    """
    处理序列输出的CWLoss变体
    输入:
        sequence_logits: (B, T, C) 序列输出 (20,1,1996)
        targets: 目标标签 (可以是标量或序列)

    返回:
        序列级别的CWLoss
    """
    B, T, C = sequence_logits.shape

    # === 目标处理 ===
    if isinstance(targets, (int, float)):
        # 标量目标 -> 扩展到整个序列
        targets = torch.full((B, T), targets, dtype=torch.float32, device=0)
    elif targets.dim() == 1:
        # 序列级目标 -> 扩展到帧级
        targets = targets.unsqueeze(1).expand(B, T)

    # === 准备目标one-hot编码 ===
    # 确保目标在有效范围内 [0, C-1]
    targets = targets.clamp(min=0, max=C - 1).long()

    # 创建目标one-hot编码 (B, T, C)
    target_one_hot = torch.zeros(B, T, C, device=0)
    target_one_hot.scatter_(2, targets.unsqueeze(-1), 1)

    # === 计算真实和最大其他分数 ===
    # 真实类别分数 (B, T)
    real = torch.sum(target_one_hot * sequence_logits, dim=-1)

    # 最大其他类别分数 (B, T)
    # 使用掩码排除目标类别
    masked_logits = sequence_logits.clone()
    masked_logits -= target_one_hot * 1e10  # 使用大负数屏蔽目标类

    other = torch.max(masked_logits, dim=-1)[0]

    # === 计算帧级损失 ===
    kappa_tensor = torch.zeros_like(real).fill_(kappa)

    if tar:
        # 目标攻击: 最小化真实分数，最大化其他分数
        frame_losses = torch.max(other - real, kappa_tensor)
    else:
        # 非目标攻击: 最大化真实分数，最小化其他分数
        frame_losses = torch.max(real - other, kappa_tensor)

    # === 序列聚合策略 ===
    # 选项1: 平均帧损失 (推荐)
    loss = torch.mean(frame_losses)

    # 选项2: 加权帧损失 (根据模型置信度)
    # probs = torch.softmax(sequence_logits, dim=-1)
    # conf_weights = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    # loss = torch.sum(frame_losses * conf_weights) / conf_weights.sum().clamp(min=1e-8)

    # 选项3: 最大帧损失 (关注最难样本)
    # loss = torch.max(frame_losses)

    return loss
class SignLanguageAligner:
    """
    手语识别专用对齐器
    - 输入: (20,1,1296) 模型输出张量
    - 标签: (9,) 标签序列
    - 输出: (20,) 对齐结果
    """

    def __init__(self, blank=0):
        self.blank = blank

    def align(self, model_output, labels):
        """
        对齐模型输出与标签序列

        参数:
            model_output: (20,1,1296) 形状的张量
            labels: (9,) 整数列表或张量

        返回:
            aligned_labels: (20,) 张量，每帧对应的标签索引
        """
        device=model_output.device
        # 转换模型输出为(T, C)形状
        logits = model_output.squeeze(1)  # 移除通道维度 -> (20, 1296)

        # 分离计算图并转换为NumPy
        log_probs = torch.log_softmax(logits.detach(), dim=-1).cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy().flatten()
        else:
            labels = np.array(labels).flatten()  # 确保形状为(9,)

        T = log_probs.shape[0]  # 时间步数 (20)
        L = len(labels)  # 标签长度 (9)

        # 处理空标签序列
        if L == 0:
            return torch.full((T,), -1, device=device)

        # 扩展标签序列（插入空白符）
        extended_labels = [self.blank]
        for l in labels:
            extended_labels.append(l)
            extended_labels.append(self.blank)
        S = len(extended_labels)  # 扩展序列长度 (19)

        # 动态规划矩阵
        dp = np.full((T, S), -np.inf)
        back_ptr = np.zeros((T, S), dtype=int)

        # 初始化
        dp[0, 0] = log_probs[0, extended_labels[0]]
        if S > 1:
            # dp[0, 1] = log_probs[0, extended_labels[1]]
            dp[0, 1] = log_probs[0, int(extended_labels[1])]

        # 前向递推
        for t in range(1, T):
            for s in range(S):
                candidates = []

                # 1. 延续当前状态
                if dp[t - 1, s] > -np.inf:
                    candidates.append((dp[t - 1, s], s))

                # 2. 从前一状态转移
                if s > 0 and dp[t - 1, s - 1] > -np.inf:
                    candidates.append((dp[t - 1, s - 1], s - 1))

                # 3. 跳过空白状态
                if (s > 1 and extended_labels[s] != self.blank and
                        extended_labels[s] != extended_labels[s - 2]):
                    if dp[t - 1, s - 2] > -np.inf:
                        candidates.append((dp[t - 1, s - 2], s - 2))

                if candidates:
                    max_val, prev_s = max(candidates, key=lambda x: x[0])
                    dp[t, s] = max_val + log_probs[t, int(extended_labels[s])]
                    back_ptr[t, s] = prev_s

        # 回溯路径
        final_state = np.argmax(dp[-1])
        path = [final_state]
        for t in range(T - 1, 0, -1):
            path.append(back_ptr[t, path[-1]])
        path.reverse()

        # 映射到原始标签索引
        alignment = []
        for s in path:
            if int(extended_labels[s]) == self.blank:
                alignment.append(-1)  # 空白符
            else:
                alignment.append((s - 1) // 2)  # 原始标签索引

        # 转换为PyTorch张量
        return torch.tensor(alignment, device=model_output.device)


# 使用示例
if __name__ == "__main__":
    # 模拟您的模型输出 (20,1,1296)
    model_output = torch.randn(20, 1, 1296).to('cuda')

    # 模拟标签序列 (9,)
    labels = np.array([10, 25, 37, 42, 58, 63, 72, 85, 96])

    # 创建对齐器
    aligner = SignLanguageAligner(blank=0)

    # 执行对齐
    aligned_labels = aligner.align(model_output, labels)

    print("输入形状:", model_output.shape)
    print("标签形状:", labels.shape)
    print("对齐结果形状:", aligned_labels.shape)
    print("对齐结果示例:")
    print(aligned_labels[:10].cpu().numpy())