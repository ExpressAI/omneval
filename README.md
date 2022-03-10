# omneval



## Install
```shell
git clone https://github.com/ExpressAI/omneval.git
cd omneval
pip install -e .
```
## Basic usage
```python
omneval ${TASKS} --arch ${ARCHS}
# e.g evaluate single task on one PLM
omneval "sst2" --arch "roberta-large"
# evaluate multiple tasks and archs at the same time, tasks and archs separated by "|"
omneval "sst2|ag_news" --archs "roberta-base|roberta-large"
```

## Available configurations

Users are free to change configurations to execute their own evaluations. There are two types of configurations you can modify:
* `task-related config`: settings about the evaluated tasks: templates, answer candidates, and proprocessing functions. 
  Most config should be fixed for fair comparison of PLMs, but you can check `omneval/tasks/{task_type}/task_zoo.py` 
  for explanation of each task-related parameters.
  
* `evaluation-related config`: task-independant settings about the evaluation process such as PLM used, output directory etc.
Check `omneval/config.py` for explanations of each evaluation-related parameters.
  
## Initiate a new task

Omneval supports tasks of different strcutures and objectives. Currently it contains seven types of tasks, each task type defines 
its input, prompt and score function. You can check the `task_zoo.py` in the task folder to see how to initialize a new task. 

## Evaluate a new PLM
Omneval supports four types of PLMs(BERT, GPT, BART, T5). If you want to evaluate a new PLM, you should categorize your 
PLM into one of the four types and register your model in `omneval/utils.py`. Currently, we support PLMs that can be 
loaded from `huggingface.transformers.Automodel.from_pretrained()` function. 

## Code Structure
Users are also free to add more tasks/task_types/evaluators for 
Three main classes: 
1. BaseConfig: a configuration class to store task-related information(datasets, prompts etc. )

2. BaseProcessor: Import, preprocess and prompt datasets

  
3. `BaseEvaluator`:recieve the`dataset` from the processor，initialize the model and conduct zero-shot evaluations. 

The definitions and attributes of three base classes can be found in `omneval/tasks/__init__.py`

