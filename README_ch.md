# CuteGPT

CuteGPT是[复旦大学知识工场实验室](http://kw.fudan.edu.cn/)推出的一个支持中英双语的开源对话语言模型，基于原版Llama进行改进和扩展，规模为13B（130亿）参数，可以在单张3090显卡上进行int8精度的推理。CuteGPT在原版Llama的基础上扩充了中文词表并进行了二次预训练，提高了对中文的理解能力，后续经过对话指令微调，提升了模型对指令的理解能力。

## 本地部署

### 安装依赖

```bash
conda create -n cutegpt python=3.7
conda activate cutegpt
pip install -r requirements.txt 
```

### 使用示例

```bash

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

预处理数据，将数据拼接成多轮对话的格式

```bash
python code/convert_data.py \
	tokenizer ziqingyang/chinese-llama-lora-7b \
	max_length 2048 \
	out_data_path data/test/
```

训练模型

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed lora_llama_flashattn.py \
    --master_port 12932 \
    --save_steps 2000 \
    --max_epoches 5 \
    --save_name llama_lora_623v1 \
    --model_path /data/xuyipei/my_llama/my_llama_13b/llama_13b_112/ \
    --dataset_type DatasetIds_HQY \
    --data_path ../weighted_dataset/623v1/llama_ift_data_ids.pkl \
    --max_length 2048 \
    --use_flash_attention
```

参数说明

- `dataset_type`：
