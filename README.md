# omneval

分为dataprocesser, evaluater 两类，base class定义在最外面的__init__.py中

新建任务直接按照task_zoo.py的样式新建config即可

Install
```shell
pip install -r requirements.txt
```
Run
```python
python main.py ${TASK} --arch ${ARCH}
# e.g
python main.py sst2 --arch roberta-large
```