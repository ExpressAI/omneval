ARCHS="bert-base-uncased|bert-large-uncased|roberta-base|roberta-large|facebook/bart-base|facebook/bart-large|openai-gpt|gpt2"
TASKS=$2

CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS}