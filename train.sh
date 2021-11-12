ARCHS="openai-gpt"
TASKS=$2

CUDA_VISIBLE_DEVICES=$1 python main.py ${TASKS} --archs ${ARCHS}