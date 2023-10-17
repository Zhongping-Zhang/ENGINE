source <path to anaconda folder>/etc/profile.d/conda.sh
conda activate <environment name for ENGINE>

python generate_articles_subset.py --model_name_or_path logs/log_gpt2_xl_goodnews_NEboth_wcap\
    --validation_file <path to goodnews folder>/goodnews_test_NE.jsonl \
    --NE_path <path to goodnews folder>/NE \
    --offset 000 \
    --dataloader "goodnews_NE_cap" \
    --p 0.96 \
    --save_interval 500 \
    
    
