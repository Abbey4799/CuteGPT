# CuteGPT

[[Readme ENG](README.md)][[Readme ZH](README_ch.md)]

CuteGPT是[复旦大学知识工场实验室](http://kw.fudan.edu.cn/)推出的一个支持中英双语的开源对话语言模型，基于原版Llama进行改进和扩展，规模为13B（130亿）参数，可以在单张3090显卡上进行int8精度的推理。CuteGPT在原版Llama的基础上扩充了中文词表并进行了二次预训练，提高了对中文的理解能力，后续经过对话指令微调，提升了模型对指令的理解能力。

## 开放参数

| Huggingface                       | 描述                                      |
| --------------------------------- | ----------------------------------------- |
| XuYipei/kw-cutegpt-13b-base       | 基于原版Llama扩充中文词表并进行二次预训练 |
| XuYipei/kw-cutegpt-13b-ift        | 全量参数指令微调                          |
| Abbey4799/kw-cutegpt-13b-ift-lora | 基于lora指令微调                          |

## 评测结果

### C-eval

#### Zero-shot

| Model                          | STEM           | Social Science | Humanities     | Other          | Average        |
| ------------------------------ | -------------- | -------------- | -------------- | -------------- | -------------- |
| GPT-4                          | 65.2           | 74.7           | 62.5           | 64.7           | 66.4           |
| ChatGPT                        | 49             | 58             | 48.8           | 50.4           | 51             |
| Claude-v1.3                    | 48.5           | 58.6           | 47.3           | 50.1           | 50.5           |
| Bloomz-mt-176B                 | 39.1           | 53             | 47.7           | 42.7           | 44.3           |
| GLM-130B                       | 36.7           | 55.8           | 47.7           | 43             | 44             |
| Claude-instant-v1.0            | 38.6           | 47.6           | 39.5           | 39             | 40.6           |
| ChatGLM-6B                     | 33.3           | 48.3           | 41.3           | 38             | 38.9           |
| LLaMA-65B                      | 32.6           | 41.2           | 34.1           | 33             | 34.7           |
| **CuteGPT-13B-ift-lora** | **30.9** | **39.3** | **37.9** | **32.4** | **34.3** |
| MOSS                           | 31.6           | 37             | 33.4           | 32.1           | 33.1           |
| Chinese-Alpaca-13B             | 27.4           | 39.2           | 32.5           | 28             | 30.9           |
| Chinese-LLaMA-13B              | 28.8           | 32.9           | 29.7           | 28             | 29.6           |

#### Five-shot

| Model                          | STEM           | Social Science | Humanities     | Other          | Average        |
| ------------------------------ | -------------- | -------------- | -------------- | -------------- | -------------- |
| GPT-4                          | 67.1           | 77.6           | 64.5           | 67.8           | 68.7           |
| ChatGPT                        | 52.9           | 61.8           | 50.9           | 53.6           | 54.4           |
| Claude-v1.3                    | 51.9           | 61.7           | 52.1           | 53.7           | 54.2           |
| Claude-instant-v1.0            | 43.1           | 53.8           | 44.2           | 45.4           | 45.9           |
| GLM-130B                       | 34.8           | 48.7           | 43.3           | 39.8           | 40.3           |
| Bloomz-mt-176B                 | 35.3           | 45.1           | 40.5           | 38.5           | 39             |
| LLaMA-65B                      | 37.8           | 45.6           | 36.1           | 37.1           | 38.8           |
| **CuteGPT-13B-ift-lora** | **33.3** | **43.1** | **40.4** | **35.5** | **37.1** |
| **CuteGPT-13B-base**     | **33.3** | **42**   | **39.7** | **33.8** | **36.4** |
| ChatGLM-6B                     | 30.4           | 39.6           | 37.4           | 34.5           | 34.5           |
| Chinese LLaMA-13B              | 31.6           | 37.2           | 33.6           | 32.8           | 33.3           |
| MOSS                           | 28.6           | 36.8           | 31             | 30.3           | 31.1           |
| Chinese Alpaca-13B             | 26             | 27.2           | 27.8           | 26.4           | 26.7           |

#### C-eval Hard

| Model                          | Zero-shot      | Five-shot      |
| ------------------------------ | -------------- | -------------- |
| GPT-4                          | 53.3           | 54.9           |
| Claude-v1.3                    | 37.6           | 39             |
| ChatGPT                        | 36.7           | 41.4           |
| Claude-instant-v1.0            | 32.1           | 35.5           |
| Bloomz-mt                      | 30.8           | 30.4           |
| GLM-130B                       | 30.7           | 30.3           |
| LLaMA-65B                      | 29.8           | 31.7           |
| **CuteGPT-13b-ift-lora** | **28.4** | **28.9** |
| **CuteGPT-13b-base**     | **N/A**  | **27.6** |
| ChatGLM-6B                     | 29.2           | 23.1           |
| MOSS                           | 28.4           | 24             |
| Chinese-LLaMA-13B              | 27.5           | 27.3           |
| Chinese-Alpaca-13B             | 24.4           | 27.1           |

### XieZhi

## 推理性能

## CuteGPT使用示例

| 任务类型 | 指令 | 标准答案 | 全量微调 | lora |
| -------- | ---- | -------- | -------- | ---- |
| 规划     | 你是一个知识图谱访问代理，你的任务是编写Python代码，使用内置的Python函数和下面给出的函数来获取用户查询相关的信息：<br>1. get_entity_info(entity_aliases)：获取一个实体的百科信息。返回'result'（实体信息或None）和'message'（描述函数调用和结果）。<br>2. find_entity_or_value(entity_aliases, relation_aliases)：找到实体或值以回答事实查询。返回'result'（实体名列表或属性值或None）和'message'（描述函数调用和结果）。<br>3. find_relationship(entity1_aliases, entity2_aliases)：预测两个实体之间的关系。返回'result'（关系或None）和'message'（描述函数调用和结果）。<br>===<br>请遵循以下规则：<br>1. 你的工作是获取相关知识，而不是直接回答查询。<br>2. 只使用内置的Python函数和提供的函数。<br>3. 在调用函数时，对实体和关系的别名进行释义和列举候选，按别名频率排序。<br>4. 在使用find_entity_or_value时，使用清晰的关系。对于模糊或广泛的关系查询，使用get_entity_info。<br>5. 通过富有逻辑的代码处理多步或嵌套的查询。<br>6. 以JSON格式响应。<br>7. 你需要一步一步思考并给出三个部分的结果：need_knowledge, thought, code。首先，你需要判断该问题是否需要知识。若是，你需要先给出一个想法，规划如何完成该查询。然后，你需要将想法转化为可执行的代码。<br>8. 所有函数调用的'messages'都记录在名为'messages'的字符串中，这是search()的返回值。<br>9. 在'messages'字符串中添加必要的解释。<br>输出格式：<br>{<br>    "need_knowledge": "<是或否>",<br>    "thought": "<你的思考过程>",<br>    "code": "def search():\n\tmessages = ''\n\t<你的代码>\n\treturn messages\n",<br>}<br>===<br>输入: 《择天记》的男主角是谁？他还有什么代表作品？  |          |          |      |
|          |      |          |          |      |
|          |      |          |          |      |
|          |      |          |          |      |
|          |      |          |          |      |
|          |      |          |          |      |
|          |      |          |          |      |
|          |      |          |          |      |

## 使用方式

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

```
bash finetune.sh
```
