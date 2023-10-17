source <path to anaconda folder>/etc/profile.d/conda.sh
conda activate <environment name for ENGINE>

python test_goodnews_NEboth_wcap.py\
     --model_name_or_path logs/log_gpt2_goodnews_NEboth_wcap \
     --validation_file <path to goodnews folder>/NE/goodnews_test_NE.jsonl \
     --NE_path <path to goodnews folder>/NE \
     --output_dir logs/log_gpt2_goodnews_NEboth_wcap/test_result \
     --do_eval \
     --per_device_eval_batch_size 8 \
     --include_art True



