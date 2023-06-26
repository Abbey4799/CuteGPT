
# CuteGPT

[[Readme ENG](README.md)][[Readme ZH](README_ch.md)]

CuteGPT is an open-source conversational language model that supports both Chinese and English, developed by [Fudan University Knowledge Workshop Laboratory](http://kw.fudan.edu.cn/). It is based on the original Llama with improvements and extensions, and has a scale of 13B (13 billion) parameters. It can perform int8 precision inference on a single 3090 graphics card. CuteGPT expands the Chinese vocabulary and performs secondary pre-training on the original Llama, improving its ability to understand Chinese. Subsequently, it is fine-tuned with conversational instructions to enhance the model's ability to understand instructions.

## Open Parameters

| Huggingface                       | Description                                                                              |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| XuYipei/kw-cutegpt-13b-base       | Expand Chinese vocabulary and perform secondary pre-training based on the original Llama |
| XuYipei/kw-cutegpt-13b-ift        | Full parameter instruction fine-tuning                                                   |
| Abbey4799/kw-cutegpt-13b-ift-lora | Instruction fine-tuning based on lora                                                    |

## Local Deployment

### Install Dependencies

```bash
conda create -n cutegpt python=3.7
conda activate cutegpt
pip install -r requirements.txt 
```

By the way, please continue with the above format, translate the following markdown text into English, wrap all your answers with ```, so I can copy them directly.

### Usage Example

```python
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch
```

* The prompt template for inference

```python
overall_instruction = "你是复旦大学知识工场实验室训练出来的语言模型CuteGPT。给定任务描述，请给出对应请求的回答。\n"
def generate_prompt(query, history, input=None):
    prompt = overall_instruction
    for i, (old_query, response) in enumerate(history):
        # Multi-turn dialogue needs to be consistent with training
        prompt += "Q: {}\nA: {}\n".format(old_query, response)
    prompt += "Q: {}\nA: ".format(query)
    return prompt
```

* Load model, tokenizer, here we use lora version of checkpoint, 8bit quantization

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

* Inference

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

You can run the following script directly for inference:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py
```

## Fine-tuning

### Data Format

The data is stored in `.pkl` format, and you can use the following code to view the data:

```python
import pickle
import random
file = 'data/test/ift_data.pkl'
with open(file, "rb") as f:
    datas = pickle.load(f)
print(len(datas))
print(random.sample(datas,3))
```

The data is stored in a two-dimensional array, where `len(datas)` represents the number of samples, and `len(data[i])` represents the number of dialogue turns for the i-th sample. Each dialogue turn consists of three key-value pairs: `instruction`, `input`, and `output`. Here is an example:

```python
[[{'instruction': '请帮我生成一首诗。', 'input': '', 'output': '好的，请给我一些关键词来启发诗歌的创作。'},
  {'instruction': '月亮、海浪、夜晚。',
   'input': '',
   'output': '心中的孤独由天空掩盖，\n夜幕笼罩下，海浪轻扬，\n月光照亮着沙滩，\n让我不再孤单。'}],
 [{'instruction': '根据给定的数据集，训练一个能够对垃圾邮件进行分类的模型。\n提供垃圾邮件和非垃圾邮件两类的数据集。',
   'input': '',
   'output': '要训练一个能...'}]]
```

### Fine-tuning Code

Preprocess the data, concatenate it into the format of multi-turn dialogues, and encode it to obtain `input_ids`, then save it.

```bash
python code/convert_data.py \
    --tokenizer ziqingyang/chinese-llama-lora-7b \
    --max_length 2048 \
    --out_data_path data/test/
```

Train the model

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

Parameter Explanation

* `model_path`: Path to the `base` model.
* `dataset_type`: Defines the `dataset` class used for data encapsulation, defined in `code/dataset.py`.
* `use_flash_attention`: Whether to use flash attention to speed up training and reduce GPU memory consumption.
* `load_lora`: Whether to load the Lora checkpoint for continued training. If `load_lora==True`, define the path to the Lora checkpoint in `load_lora_path`.

Refer to `code/config.py` for specific deepspeed parameters (e.g., learning rate, batch size) and Lora parameters (e.g., Lora rank).

You can directly run the following command to start training:

```
bash finetune.sh
```
