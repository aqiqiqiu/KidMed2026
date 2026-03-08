import pickle
from transformers import BertTokenizerFast
from parameter_config import ParameterConfig
import os
config = ParameterConfig()

# 加载tokenizer
vocab_dir = os.path.dirname(config.vocab_path)
tokenizer = BertTokenizerFast.from_pretrained(
    vocab_dir,
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",

)

# 加载原数据
with open(config.train_path, 'rb') as f:
    raw_data = pickle.load(f)

# 重新格式化为对话格式
new_data = []
for item in raw_data:
    if isinstance(item, list):
        # 如果是token IDs，先解码
        text = tokenizer.decode(item, skip_special_tokens=True)
    else:
        text = item

    # 假设格式是 "问题 回答"
    if "？" in text or "?" in text:
        parts = text.split("？")
        if len(parts) == 2:
            question = parts[0] + "？"
            answer = parts[1]
        else:
            parts = text.split("?")
            if len(parts) == 2:
                question = parts[0] + "？"
                answer = parts[1]
            else:
                # 无法分割，跳过或默认处理
                continue
    else:
        # 如果没有问号，可能是单句，跳过
        continue

    # 格式化为对话
    dialogue = f"用户:{question} 助手:{answer}"
    tokens = tokenizer.encode(dialogue, add_special_tokens=True)
    new_data.append(tokens)

print(f"原数据: {len(raw_data)} 条")
print(f"新数据: {len(new_data)} 条")

# 保存新数据
new_path = config.train_path.replace('.pkl', '_dialog.pkl')
with open(new_path, 'wb') as f:
    pickle.dump(new_data, f)

print(f"新数据已保存到: {new_path}")