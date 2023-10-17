source <path to anaconda folder>/etc/profile.d/conda.sh
conda activate <environment name for ENGINE>

export PYTHONPATH=$PWD
python test_goodnews.py\
     --model_name_or_path logs/log_gpt2_goodnews\
     --validation_file <path to goodnews folder>/goodnews_test.jsonl \
     --output_dir logs/log_gpt2_goodnews/test_result \
     --do_eval \
     --per_device_eval_batch_size 8


