
function eval_ml() {
  language=$1
  echo "Current input language is: " $language
  echo "Current used model is: " $2
  echo "Current used cuda is: " $3
  echo "Current max span length is: " $4
  data_name="panx"
  path_temp="./data/ner/multilingual_NER_template.xlsx"
  dir="./data/ner/panx/"
  column_no=1
  delimiter="\t"
  max_span_width=$4
  eval_model_class="mbart"
  eval_model=$2
  eval_tokenizer=$eval_model
  score_fun="TNSAT"
  temp_form="generate"
  use_small_data=True # test 10 samples randomly
  cuda=$3
  eval_model1=${eval_model/\//\\}
  model_name=${data_name}_${language}_${eval_model1}_mspan${max_span_width}_score${score_fun}_${temp_form}_small${use_small_data}

  CUDA_VISIBLE_DEVICES=$3 nohup python prompt_ner.py \
  --data_name  $data_name \
  --path_temp $path_temp \
  --dir $dir \
  --language $language \
  --column_no $column_no \
  --max_span_width $max_span_width \
  --eval_model_class $eval_model_class \
  --eval_model $eval_model \
  --eval_tokenizer $eval_tokenizer \
  --score_fun $score_fun \
  --temp_form $temp_form \
  --use_small_data $use_small_data \
  --model_name $model_name \
  >logs/${model_name}_1101_2021.txt &
}

cuda=0

#language='ja' # zh
#max_span_width=20
#max_span_width=20 #{zh, } character-level need to set a long span length, such as 20

max_span_width=6
language='en' # es, de, el, en, fr, it
models=("facebook/mbart-large-50")


for model in ${models[*]}; do
  echo $language
  echo $model
  eval_ml $language $model $cuda $max_span_width

done




