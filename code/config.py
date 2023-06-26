
from peft import LoraConfig


DS_CONFIG = {
    "bf16": {
        "enabled": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.98, 0.999],
            "eps": 1e-9
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-4,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 300
        }
    },
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        # "offload_optimizer": {
        #     "device": "cpu"
        # },
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": False
}

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)



input_template_pool = [
    "Input:{}",
    "Question:{}",
    "Problem:{}",
    "Q:{}",
    "*Question*:{}",
    "(Problem){}",
    "{}",
    "{}",
    "{}"
]

template_pool = {
    'wround_woinput':[
            "问：{}\n答：{}\n",
        "Instruction:{}\Response:{}\n",
        "{}\n{}\n"
    ],
    'wround_winput':[
         "背景：{}\n{}\n答：{}\n",
          "已知：{}\n{}\n回答：{}\n",
            "问：{}\n{}\n答：{}\n",
        "Instruction:{}\n{}\nResponse:{}\n",
        "{}\n{}\n{}\n"
    ],
    'woround_woinput':[
                "问：{}\n答：{}\n",
        "Instruction:{}\nResponse:{}\n"
        "{}\n{}\n"
    ],
    'woround_winput':[
                "问：{}\n{}\n答：{}\n",
        "Instruction:{}\n{}\nResponse:{}\n"
        "{}\n{}\n{}\n"
    ]
}
meta_prompt = '你是复旦大学知识工场实验室训练出来的语言模型CuteGPT。给定任务描述，请给出对应请求的回答。\n'