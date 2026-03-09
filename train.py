import sys
import os
import torch
from datetime import datetime
import transformers
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Model, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast
from torch.optim import AdamW
from function_tools import *  # 注意：是 functions_tools 不是 function_tools
from parameter_config import *
from data_handle.dataloader import get_loader


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch, config):
    model.train()
    model.to(config.device)
    ignore_index = config.ignore_index
    loss_sum = 0
    correct = 0
    total_words = 0

    print(f"开始训练 Epoch {epoch + 1}")

    for idx, (input_id, labels) in enumerate(train_dataloader):
        input_ids = input_id.to(config.device)
        labels = labels.to(config.device)

        outputs = model(input_ids, labels=labels)
        logits = outputs.logits
        loss = outputs.loss

        n_c, n_word = calculate_acc(logits, labels, ignore_index)
        correct += n_c
        total_words += n_word

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        loss.backward()

        if (idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_sum += loss.item() * config.gradient_accumulation_steps

        if (idx + 1) % 100 == 0:
            avg_loss = loss_sum / (idx + 1)
            avg_acc = correct / total_words if total_words > 0 else 0
            print(f"  Batch {idx + 1}/{len(train_dataloader)}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    epoch_loss = loss_sum / len(train_dataloader)
    epoch_acc = correct / total_words if total_words > 0 else 0

    print(f"Epoch {epoch + 1} 完成, 平均 Loss: {epoch_loss:.4f}, 平均 Acc: {epoch_acc:.4f}")

    # 保存模型
    print(f'正在保存第 {epoch + 1} 轮的模型...')
    epoch_start_time = datetime.now()
    model_path = os.path.join(config.save_model_path, f'bj_epoch{epoch + 1}')
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    print(f'第 {epoch + 1} 轮模型保存完成')
    epoch_finish_time = datetime.now()
    print(f'本轮训练时间: {epoch_finish_time - epoch_start_time}')

    return epoch_loss


def validate_epoch(model, validate_dataloader, epoch, config):
    print("开始验证...")
    model.eval()
    device = config.device
    ignore_index = config.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # 删除这里的 break！

    epoch_mean_loss = total_loss / len(validate_dataloader)
    print(f"validate epoch {epoch + 1}: loss {epoch_mean_loss:.6f}")
    epoch_finish_time = datetime.now()
    print(f'验证时间: {epoch_finish_time - epoch_start_time}')
    return epoch_mean_loss


def train(model, train_dataloader, valid_dataloader, config):
    # 计算总训练步数
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.epochs

    optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.eps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=t_total
    )

    train_loss_list = []
    valid_loss_list = []
    valid_loss_best = float('inf')

    print(f"开始训练，共 {config.epochs} 轮")
    print(f"训练数据批次: {len(train_dataloader)}")
    print(f"总优化步数: {t_total}")
    print("=" * 50)

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 30)

        # 训练
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, epoch, config
        )
        train_loss_list.append(train_loss)
        print(f"训练 Loss: {train_loss:.6f}")

        # 验证
        valid_loss = validate_epoch(
            model, valid_dataloader, epoch, config
        )
        valid_loss_list.append(valid_loss)
        print(f"验证 Loss: {valid_loss:.6f}")

        # 保存最佳模型
        if valid_loss < valid_loss_best:
            valid_loss_best = valid_loss
            print(f"发现更好的模型，验证 Loss 降到 {valid_loss:.6f}")
            os.makedirs(config.save_model_path, exist_ok=True)
            model.save_pretrained(config.save_model_path)
            print(f"模型已保存到 {config.save_model_path}")

        # 每5轮打印总结
        if (epoch + 1) % 5 == 0:
            avg_train = sum(train_loss_list[-5:]) / 5
            avg_valid = sum(valid_loss_list[-5:]) / 5
            print(f"\n最近5轮平均 - 训练 Loss: {avg_train:.6f}, 验证 Loss: {avg_valid:.6f}")

    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳验证 Loss: {valid_loss_best:.6f}")
    return train_loss_list, valid_loss_list


def main():
    config = ParameterConfig()

    # 1. 分词器 - 正确初始化
    vocab_dir = os.path.dirname(config.vocab_path)
    tokenizer = BertTokenizerFast.from_pretrained(
        vocab_dir,
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]"
    )
    print(f"词汇表大小: {len(tokenizer)}")  # 现在应该是13317

    # 2. 模型
    if config.pretrained_model and os.path.exists(config.pretrained_model):
        print(f"加载预训练模型: {config.pretrained_model}")
        model = GPT2LMHeadModel.from_pretrained(config.pretrained_model)
    else:
        print("创建新模型")
        conf = GPT2Config.from_json_file(config.config_json)
        model = GPT2LMHeadModel(config=conf)

    model.to(config.device)
    print(f"模型词汇表大小: {model.config.vocab_size}")  # 应该是13317

    # 验证两个词汇表是否匹配
    if len(tokenizer) != model.config.vocab_size:
        print(f"警告：tokenizer词汇表({len(tokenizer)})和模型词汇表({model.config.vocab_size})不匹配！")

    # 3. 数据加载
    print("加载数据...")
    train_dataloader, valid_dataloader = get_loader(
        train_path=config.train_path,
        valid_path=config.valid_path
    )
    print(f"训练数据批次: {len(train_dataloader)}")
    print(f"验证数据批次: {len(valid_dataloader)}")

    # 4. 训练
    train(model, train_dataloader, valid_dataloader, config)


if __name__ == '__main__':
    main()