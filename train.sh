#ARCHS="bert-base-uncased|bert-large-uncased|roberta-large|roberta-base|facebook/bart-base|facebook/bart-large|t5-base|t5-large|openai-gpt|gpt2"
#ARCHS="t5-base|t5-large|sshleifer/distilbart-cnn-12-6"
#ARCHS="openai-gpt|t5-large"

#ARCHS="facebook/bart-base|facebook/bart-large|t5-base|t5-large|openai-gpt|gpt2"
#ARCHS="facebook/bart-base|facebook/bart-large|t5-base|t5-large|facebook/bart-large-cnn|sshleifer/distilbart-cnn-12-6"
#ARCHS="t5-large|facebook/bart-large-cnn|sshleifer/distilbart-cnn-12-6"
#ARCHS="openai-gpt|sshleifer/distilbart-cnn-12-6"
#ARCHS="bert-large-uncased"

#ARCHS="roberta-large|roberta-base|facebook/bart-base|facebook/bart-large|t5-base|t5-large|facebook/bart-large-cnn|sshleifer/distilbart-cnn-12-6|gpt2|facebook/bart-large-xsum|facebook/bart-large-mnli"
#ARCHS="bert-base-uncased|bert-large-uncased|roberta-large|roberta-base|facebook/bart-base|facebook/bart-large|facebook/bart-large-cnn|sshleifer/distilbart-cnn-12-6"
#ARCHS="gpt2"
ARCHS='t5-11b|t5-base|t5-large'
#ARCHS="facebook/bart-large-mnli|facebook/bart-large-xsum|facebook/bart-base|facebook/bart-large|facebook/bart-large-cnn|t5-base|t5-large"
#ARCHS="facebook/bart-large-xsum"
#ARCHS="Michau/t5-base-en-generate-headline|mrm8488/t5-base-finetuned-common_gen"
#ARCHS="textattack/bert-base-uncased-imdb"
#ARCHS="deepset/roberta-base-squad2"
#ARCHS="deepset/bert-base-uncased-squad2"
#ARCHS="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli|finiteautomata/bertweet-base-sentiment-analysis|siebert/sentiment-roberta-large-english|textattack/bert-base-uncased-MN"
TASKS=$2
CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --meta_prefix meta_test

#CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS}  --output_inference

#CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS} --meta_prefix meta_expand