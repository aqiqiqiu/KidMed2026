# flask_predict.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 使用刚下载的模型路径
LOCAL_MODEL_PATH = "./qwen2.5-1.5b-instruct"

print(f"从本地加载模型: {LOCAL_MODEL_PATH}")
print("正在加载tokenizer...")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    padding_side="right"
)

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在加载模型（这可能需要一点时间）...")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    local_files_only=True,
    low_cpu_mem_usage=True
)

model.eval()
print("✅ 模型加载完成！")

# 预设回答库
medical_responses = {
    "你好": "您好！我是儿童医疗助手，有什么可以帮助您的？",
    "头疼": "小朋友头疼可能由多种原因引起，建议先休息，如果持续疼痛请及时就医。",
    "头痛": "头疼可能由多种原因引起，建议先休息，如果持续疼痛请及时就医。",
    "感冒": "感冒建议多喝水、多休息。如果发烧超过38.5℃可以服用退烧药。",
    "发烧": "请测量体温，38.5℃以下可物理降温，以上可服用退烧药。建议咨询医生。",
    "咳嗽": "是干咳还是有痰？持续多久了？建议多喝温水，避免吃刺激性食物。",
    "胃疼": "胃疼可能与饮食有关，建议饮食清淡，避免辛辣刺激食物。",
    "嗓子疼": "嗓子疼可能是咽喉炎，多喝温水，少说话。",
}


def model_predict(text):
    """使用本地Qwen模型生成回答"""

    # 先快速匹配预设回答
    for key in medical_responses:
        if key in text:
            return medical_responses[key]

    try:
        # 构建对话消息
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的儿童医疗助手，专门为家长解答儿童健康问题。请用温和、易懂的语气回答，既要专业又要让家长放心。"
            },
            {"role": "user", "content": text}
        ]

        # 应用聊天模板
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # 移到GPU（如果有）
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 解码回复
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        response = response.strip()
        return response if len(response) > 5 else "请详细描述小朋友的症状。"

    except Exception as e:
        print(f"生成出错: {e}")
        return "抱歉，我现在有点忙，请稍后再试。"