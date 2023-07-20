from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch

overall_instruction = "你是复旦大学知识工场实验室训练出来的语言模型CuteGPT。给定任务描述，请给出对应请求的回答。\n"
def generate_prompt(query, history, input=None):
    prompt = overall_instruction
    for i, (old_query, response) in enumerate(history):
        # 多轮对话需要跟训练时保持一致
        prompt += "问：{}\n答：\n{}\n".format(old_query, response)
    prompt += "问：{}\n答：\n".format(query)
    return prompt

model_name = "XuYipei/kw-cutegpt-13b-base"
LORA_WEIGHTS = "Abbey4799/kw-cutegpt-13b-ift-lora"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
device = torch.device("cuda")

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
