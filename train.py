import os
import torch
import pickle
from datetime import datetime
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, DataLoader
from parameter_config import ParameterConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 加载数据
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        self.examples = []
        for item in raw_data:
            if isinstance(item, list):
                # 如果是token IDs，先解码
                text = tokenizer.decode(item, skip_special_tokens=True)
            else:
                text = item

            # 转换为对话格式
            if "用户:" in text or "助手:" in text:
                # 已经是对话格式
                self.examples.append(text)
            else:
                # 尝试解析为问答对
                self._parse_qa_pair(text)

        logger.info(f"加载了 {len(self.examples)} 条训练数据")

    def _parse_qa_pair(self, text):
        """解析问答对为对话格式"""
        # 简单的启发式分割
        if "？" in text or "?" in text:
            parts = text.split("？" if "？" in text else "?")
            if len(parts) >= 2:
                question = parts[0] + "？"
                answer = "？".join(parts[1:])  # 可能有多个部分

                # 格式化为对话
                dialogue = f"用户：{question}\n助手：{answer}"
                self.examples.append(dialogue)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Qwen的对话格式
        messages = [
            {"role": "user", "content": text.split("助手：")[0].replace("用户：", "").strip()},
            {"role": "assistant", "content": text.split("助手：")[1].strip()}
        ]

        # 应用聊天模板
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],  # 只有用户消息
            tokenize=False,
            add_generation_prompt=True
        )
        full_text = prompt + messages[-1]["content"] + self.tokenizer.eos_token

        # 编码
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None
        )

        input_ids = encodings["input_ids"]

        # 创建标签（只计算助手部分的损失）
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }


def collate_fn(batch):
    """动态padding"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # 找到最大长度
    max_len = max(len(ids) for ids in input_ids)

    # padding
    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for ids, lbls in zip(input_ids, labels):
        padding_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbls, torch.zeros(padding_len, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(ids)), torch.zeros(padding_len)]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks)
    }


def train():
    config = ParameterConfig()

    # 加载tokenizer和模型
    logger.info(f"加载模型: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # 加载数据集
    logger.info("加载训练数据")
    train_dataset = MedicalDataset(config.train_path, tokenizer, config.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.eps)

    # 总训练步数
    t_total = len(train_loader) * config.epochs // config.gradient_accumulation_steps

    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=t_total
    )

    # 训练
    logger.info("开始训练")
    model.train()
    global_step = 0
    total_loss = 0

    for epoch in range(config.epochs):
        epoch_loss = 0
        for step, batch in enumerate(train_loader):
            # 移至设备
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # 梯度累积
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            epoch_loss += loss.item()
            total_loss += loss.item()

            # 更新参数
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # 打印进度
            if step % 50 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs}, Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # 每个epoch结束后保存模型
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} 完成，平均 Loss: {avg_epoch_loss:.4f}")

        # 保存模型
        save_path = os.path.join(config.save_model_path, f'epoch_{epoch + 1}')
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"模型已保存到 {save_path}")

    # 保存最终模型
    model.save_pretrained(config.save_model_path)
    tokenizer.save_pretrained(config.save_model_path)
    logger.info(f"训练完成，最终模型保存到 {config.save_model_path}")


if __name__ == "__main__":
    train()