# CuteGPT

[[Readme ENG](README.md)][[Readme ZH](README_ch.md)]

CuteGPT是[复旦大学知识工场实验室](http://kw.fudan.edu.cn/)推出的一个支持中英双语的开源对话语言模型，基于原版Llama进行改进和扩展，规模为13B（130亿）参数，可以在单张3090显卡上进行int8精度的推理。CuteGPT在原版Llama的基础上扩充了中文词表并进行了二次预训练，提高了对中文的理解能力，后续经过对话指令微调，提升了模型对指令的理解能力。

## 开放参数

| Huggingface                       | 描述                                      |
| --------------------------------- | ----------------------------------------- |
| XuYipei/kw-cutegpt-13b-base       | 基于原版Llama扩充中文词表并进行二次预训练 |
| XuYipei/kw-cutegpt-13b-ift        | 全量参数指令微调                          |
| Abbey4799/kw-cutegpt-13b-ift-lora | 基于lora指令微调                          |

## 本地部署

### 安装依赖

```bash
conda create -n cutegpt python=3.7
conda activate cutegpt
pip install -r requirements.txt 
```

### 使用示例

```python
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch
```

- 推理时的prompt模版

```bash
overall_instruction = "你是复旦大学知识工场实验室训练出来的语言模型CuteGPT。给定任务描述，请给出对应请求的回答。\n"
def generate_prompt(query, history, input=None):
    prompt = overall_instruction
    for i, (old_query, response) in enumerate(history):
        # 多轮对话需要跟训练时保持一致
        prompt += "问：{}\n答：{}\n".format(old_query, response)
    prompt += "问：{}\n答：".format(query)
    return prompt
```

- 加载模型、分词器，此处采用lora版本的checkpoint、8bit量化

```python
model_name = "XuYipei/kw-cutegpt-13b-base"
LORA_WEIGHTS = "Abbey4799/kw-cutegpt-13b-ift-lora"
tokenizer = LlamaTokenizer.from_pretrained(LORA_WEIGHTS)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
device = torch.device("cuda")

```

- 推理

```python
history = []
queries = ['请推荐五本名著，依次列出作品名、作者','再来三本呢？']
memory_limit = 3 # the number of (query, response) to remember
for query in queries:
    prompt = generate_prompt(query, history)
    print(prompt)

    input_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
    input_ids = input_ids["input_ids"].to(device)

    with torch.no_grad():
        outputs=model.generate(
                input_ids=input_ids,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.1,
                max_new_tokens = 256,
                early_stopping = True,
                eos_token_id = tokenizer.convert_tokens_to_ids('<s>'),
                pad_token_id = tokenizer.eos_token_id,
                min_length = input_ids.shape[1] + 1
        )
    s = outputs[0][input_ids.shape[1]:]
    response=tokenizer.decode(s)
    response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '')
    print(response)
    history.append((query, response))
    history = history[-memory_limit:]
```

可以直接运行以下脚本进行推理：

```python
CUDA_VISIBLE_DEVICES=0 python inference.py
```

## 微调

### 数据格式

数据以 `.pkl`的形式存储，可以用以下代码查看数据：

```python
import pickle
import random
file = 'data/test/ift_data.pkl'
with open(file, "rb") as f:
    datas = pickle.load(f)
print(len(datas))
print(random.sample(datas,3))
```

数据由二维数组保存，`len(datas)`为样本的数量，`len(data[i])`代表第 `i`个样本的对话轮数，其中每轮对话由 `instruction`、`input`、`output`三个键值构成。示例如下：

```python
[[{'instruction': '请帮我生成一首诗。', 'input': '', 'output': '好的，请给我一些关键词来启发诗歌的创作。'},
  {'instruction': '月亮、海浪、夜晚。',
   'input': '',
   'output': '心中的孤独由天空掩盖，\n夜幕笼罩下，海浪轻扬，\n月光照亮着沙滩，\n让我不再孤单。'}],
 [{'instruction': '根据给定的数据集，训练一个能够对垃圾邮件进行分类的模型。\n提供垃圾邮件和非垃圾邮件两类的数据集。',
   'input': '',
   'output': '要训练一个能...'}]]
```

### 微调代码

预处理数据，将数据拼接成多轮对话的格式，并编码得到 `input_ids`，存储下来

```bash
python code/convert_data.py \
	--tokenizer ziqingyang/chinese-llama-lora-7b \
	--max_length 2048 \
	--out_data_path data/test/
```

训练模型

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 12932 code/finetune.py \
    --save_steps 2000 \
    --max_epoches 4 \
    --save_name llama_lora \
    --model_path XuYipei/kw-cutegpt-13b-base \
    --dataset_type DatasetIds \
    --data_path data/test/llama_ift_data_ids.pkl \
    --max_length 2048 \
    --use_flash_attention
```

参数说明

- `model_path`：`base`模型的路径
- `dataset_type`：封装数据的 `dataset`类定义，在 `code/dataset.py`中定义
- `use_flash_attention`：是否使用flash attention加快训练、减少显存消耗
- `load_lora`：是否读取lora checkpoint继续训练。如果 `load_lora==True`，在 `load_lora_path`中定义lora checkpoint的路径

具体的 deepspeed 参数（例如 ` learning rate`、` batch size`）以及   `lora `参数（例如 ` lora rank`）见  ` code/config.py`

可以直接运行以下指令进行训练：

```python
bash finetune.sh
```
