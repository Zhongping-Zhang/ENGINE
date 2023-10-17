source <path to anaconda folder>/etc/profile.d/conda.sh
conda activate <environment name for ENGINE>

python goodnews_gpt2_NEsec_wcap.py \
     --model_name_or_path gpt2 \
     --train_file <path to goodnews folder>/NE/goodnews_train_NE.jsonl \
     --validation_file <path to goodnews folder>/NE/goodnews_val_NE.jsonl \
     --NE_path <path to goodnews folder>/NE \
     --output_dir logs/log_gpt2_goodnews_NEboth_wcap \
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
     --metric_for_best_model eval_loss \
     --include_art True


