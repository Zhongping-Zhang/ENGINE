source <path to anaconda folder>/etc/profile.d/conda.sh
conda activate <environment name for ENGINE>

export PYTHONPATH=$PWD
python goodnews_gpt2.py \
     --model_name_or_path gpt2 \
     --train_file <path to goodnews folder>/goodnews_train.jsonl \
     --validation_file <path to goodnews folder>/goodnews_val.jsonl \
     --output_dir logs/log_gpt2_goodnews \
     --do_train \
     --do_eval \
     --evaluation_strategy steps \
     --eval_steps 2000 \
     --per_device_train_batch_size 8 \
     --per_device_eval_batch_size 8 \
     --learning_rate 1e-4 \
     --max_steps 100000 \
     --save_steps 2000 \
     --save_total_limit 5 \
     --warmup_steps 1000 \
     --load_best_model_at_end \
     --metric_for_best_model eval_loss

# model architecture: [gpt2, gpt2-medium, gpt2-xl]

