CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 12932 code/finetune.py \
    --save_steps 2000 \
    --max_epoches 4 \
    --save_name llama_lora \
    --model_path XuYipei/kw-cutegpt-13b-base \
    --dataset_type DatasetIds \
    --data_path data/test/llama_ift_data_ids.pkl \
    --max_length 2048 \
    --use_lora \
    --use_flash_attention