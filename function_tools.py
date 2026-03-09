# loss acc
# loss acc

import torch
import torch.nn.functional as F

def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[:, :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[:, 1:].contiguous().view(-1)
    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    '''
    在 PyTorch 中，labels.ne(ignore_index) 表示将标签张量 labels 中的值不等于 ignore_index 的位置标记为 True，等于 ignore_index 的位置标记为 False。
    这个操作通常被用于计算交叉熵损失，以过滤掉 ignore_index 对损失的贡献
    '''
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    '''
    在 PyTorch 中，
    logit.eq(labels) 表示将模型的预测输出值 logit 中等于标签张量 labels 的位置标记为 True，不等于标签张量 labels 的位置标记为 False。这个操作通常被用于计算交叉熵损失，以标记出预测输出值和标签值相等的位置。
    masked_select(non_pad_mask) 表示将张量中非填充标记的位置选出来。这个操作通常被用于计算损失时，过滤掉填充标记对损失的影响。
    '''
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word

# loss未使用，只是介绍损失计算的过程
def caculate_loss(logit, target, pad_idx, smoothing=False):
    '''
    计算模型的损失：通过函数解析下，GPT2内部如何计算损失的
    :param logit: 模型预测结果
    :param target: 真实标签
    :param pad_idx:特殊-100忽略计算损失的值
    :param smoothing: 不是核心，是计算损失的优化方法
    :return:
    '''
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)
        #
        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        """
        logits: 精神症状EOS
        label: 精神症状EOS
        """

        ###
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss