export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=0

DATASET=Instruments
BASE_MODEL=./llama-7b
ITEM_MODEL=./best_collision_model.pth
USER_MODEL=./best_collision_model.pth
DATA_PATH=./data
OUTPUT_DIR=./$DATASET

torchrun --nproc_per_node=8 pre-train.py \
    --base_model $BASE_MODEL \
    --item_model $ITEM_MODEL \
    --user_model $USER_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --deepspeed ./config/ds_z2_fp16.json \
    --dataloader_num_workers 4 \
    --only_train_response \
    --tasks seqrec,itemsearch,inters2title,inters2description,preferenceobtain,item2index,index2item,intertitles2item,query2item,usersearch,user2pref,pref2user \
    --train_prompt_sample_num 1,1,1,1,1,1,1,1,1,1,1,1 \
    --train_data_sample_num 0,0,0,0,0,0,0,0,0,0,0,0 \
    --index_file .index.json \
    --user_index_file .user-index.json \
    --fp16

cd convert
nohup ./convert.sh $OUTPUT_DIR >convert.log 2>&1 &
cd ..

CKPT_PATH=$OUTPUT_DIR

python generate_embeddings.py \
    --ckpt_path $CKPT_PATH \
    --item_save_path $CKPT_PATH/embeddings.item.tsv \
    --user_save_path $CKPT_PATH/embeddings.user.tsv \
    --device_map 0

python generate_indices.py \
    --ckpt_path $CKPT_PATH \
    --item_data_path $CKPT_PATH/embeddings.item.tsv \
    --user_data_path $CKPT_PATH/embeddings.user.tsv \
    --save_path $CKPT_PATH \
    --device_map 0

export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=0

DATASET=Instruments
CKPT_PATH=$OUTPUT_DIR
DATA_PATH=./data
OUTPUT_DIR=$CKPT_PATH/finetune

torchrun --nproc_per_node=8 fine-tune.py \
    --ckpt_path $CKPT_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --epochs 4 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --fp16 \
    --deepspeed ./config/ds_z2_fp16.json \
    --dataloader_num_workers 4 \
    --only_train_response \
    --tasks seqrec,itemsearch,preferenceobtain,item2index,index2item,fusionseqrec,usersearch,user2pref,pref2user \
    --train_prompt_sample_num 1,1,1,1,1,1,1,1,1 \
    --train_data_sample_num 0,0,0,0,0,0,0,0,0 \
    --index_file $CKPT_PATH/indices.item.json \
    --user_index_file $CKPT_PATH/indices.user.json \
    --reindex 17 

cd convert
nohup ./convert.sh $OUTPUT_DIR >convert.log 2>&1 &
cd ..

DATASET=Instruments
BASE_MODEL=./llama-7b
DATA_PATH=./data
INDEX_PATH=$CKPT_PATH
CKPT_PATH=$OUTPUT_DIR
RESULTS_FILE=$CKPT_PATH/eval_result.json

torchrun --nproc_per_node=8 evaluate-finetuned.py \
    --base_model $BASE_MODEL \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 10 \
    --test_prompt_ids all \
    --test_task seqrec \
    --index_file $INDEX_PATH/indices.item.json \
    --user_index_file $INDEX_PATH/indices.user.json