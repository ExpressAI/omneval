# omneval



## Install
```shell
git clone https://github.com/ExpressAI/omneval.git
cd omneval
pip install -e .
```
## Basic usage
```python
omneval ${TASK} --arch ${ARCH}
# e.g
omneval sst2 --arch roberta-large
```

## Available configurations
### task-related configurations
check files in `omneval/tasks/{task_type}/task_zoo.py` for explanation of each task-related parameters.

An example of task configurations can be found below. 

### evaluation-related configurations

For evaluation-related configurations(PLMs, output_dir etc.), pleaqse check `omneval/tasks/{task_type}/task_zoo.py` for details. 

## Code Structure
Three main classes: 
1. BaseConfig: a configuration class to store task-related information(datasets, prompts etc. )

2. BaseProcessor: Import, preprocess and prompt datasets

  
3. `BaseEvaluator`:recieve the`dataset` from the processor，initialize the model and conduxt zero-shot evaluations. 

The definitions and attributes of three base classes can be found in `omneval/tasks/__init__.py`


### To be continued 