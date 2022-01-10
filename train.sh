#ARCHS="bert-base-uncased|bert-large-uncased|roberta-large|roberta-base|facebook/bart-base|facebook/bart-large|t5-base|t5-large"
#ARCHS="t5-base|t5-large|sshleifer/distilbart-cnn-12-6"
#ARCHS="bert-base-uncased|bert-large-uncased"

ARCHS="bert-base-uncased|bert-large-uncased|roberta-large|roberta-base|facebook/bart-base|facebook/bart-large|t5-base|t5-large|facebook/bart-large-cnn|sshleifer/distilbart-cnn-12-6|openai-gpt|gpt2"
#ARCHS="facebook/bart-base|facebook/bart-large|t5-base|t5-large|facebook/bart-large-cnn|sshleifer/distilbart-cnn-12-6"
#ARCHS="t5-large|facebook/bart-large-cnn|sshleifer/distilbart-cnn-12-6"
#ARCHS="openai-gpt|gpt2"
#ARCHS="bert-base-uncased|bert-large-uncased|roberta-large|roberta-base"

#ARCHS="openai-gpt"

TASKS=$2
#CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --output_inference --calibrate --meta_prefix meta_calibrate

CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --output_inference

#CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --meta_prefix meta_expand