# flask_predict.py
import torch
from transformers import GPT2LMHeadModel, BertTokenizerFast
from parameter_config import ParameterConfig
import os

# 加载配置
config = ParameterConfig()

# 加载tokenizer
vocab_dir = os.path.dirname(config.vocab_path)
tokenizer = BertTokenizerFast.from_pretrained(
    vocab_dir,
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]"
)

# 加载模型
model = GPT2LMHeadModel.from_pretrained(config.save_model_path)
model.eval()

# 预设回答库
medical_responses = {
    "你好": "您好！我是医疗助手，有什么可以帮助您的？",
    "头疼": "头疼可能由多种原因引起，建议您先休息，如果持续疼痛请及时就医。",
    "头痛": "头疼可能由多种原因引起，建议您先休息，如果持续疼痛请及时就医。",
    "感冒": "感冒建议多喝水、多休息。如果发烧超过38.5℃可以服用退烧药。",
    "发烧": "请测量体温，38.5℃以下可物理降温，以上可服用退烧药。",
    "咳嗽": "是干咳还是有痰？持续多久了？建议多喝温水。",
    "胃疼": "胃疼可能与饮食有关，建议饮食清淡，避免辛辣刺激食物。",
    "胃痛": "胃疼可能与饮食有关，建议饮食清淡，避免辛辣刺激食物。",
    "嗓子疼": "嗓子疼可能是咽喉炎，多喝温水，少说话。",
    "喉咙痛": "嗓子疼可能是咽喉炎，多喝温水，少说话。",
}


def clean_response(response):
    """清理模型生成的回答"""
    parts = response.split("助手:")
    if len(parts) > 1:
        response = parts[-1].strip()
    response = response.replace(" ", "")
    return response


def model_predict(text):
    # 1. 先检查预设回答
    for key in medical_responses:
        if key in text:
            return medical_responses[key]

    # 2. 用模型生成
    prompt = f"用户:{text} 助手:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 40,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
            num_return_sequences=3
        )

    # 选最好的回答
    for output in outputs:
        full = tokenizer.decode(output, skip_special_tokens=True)
        response = clean_response(full)
        if len(response) > 5 and "，" in response:
            return response

    # 3. 如果模型生成不好，返回通用回答
    return "请详细描述您的症状，我会尽力帮您分析。建议您咨询专业医生。"