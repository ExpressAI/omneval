ARCHS="bert-base-uncased|bert-large-uncased|roberta-base|roberta-large|facebook/bart-base|facebook/bart-large"
TASKS=$1

python main.py ${TASKS} --archs ${ARCHS}