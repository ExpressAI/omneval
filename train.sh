#ARCHS="bert-base-uncased|bert-large-uncased|roberta-large|roberta-base|facebook/bart-base|facebook/bart-large|t5-base|t5-large"
ARCHS="roberta-large|roberta-base"
#ARCHS="t5-base|t5-large"

ARCHS="facebook/bart-base|facebook/bart-large"

TASKS=$2
#CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --output_inference --calibrate --meta_prefix meta_calibrate

CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --output_inference