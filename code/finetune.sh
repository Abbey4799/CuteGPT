CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed lora_llama_flashattn.py \
    --master_port 12932 \
    --save_steps 2000 \
    --max_epoches 5 \
    --save_name llama_lora_623v1 \
    --model_path /data/xuyipei/my_llama/my_llama_13b/llama_13b_112/ \
    --dataset_type DatasetIds_HQY \
    --data_path ../weighted_dataset/623v1/llama_ift_data_ids.pkl \
    --max_length 2048