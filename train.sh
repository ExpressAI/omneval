#ARCHS="bert-base-uncased|bert-large-uncased|roberta-large|roberta-base|facebook/bart-base|facebook/bart-large|t5-base|t5-large"
ARCHS="bert-base-uncased|bert-large-uncased|roberta-large|roberta-base|facebook/bart-base|facebook/bart-large|openai-gpt|gpt2"
#ARCHS="t5-base|t5-large"

#ARCHS="facebook/bart-large|facebook/bart-base|facebook/bart-large-cnn"

TASKS=$2
#CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --output_inference --calibrate --meta_prefix meta_calibrate

CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --output_inference