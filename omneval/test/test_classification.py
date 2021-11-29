import unittest
import subprocess
import re
import os
from omneval.tasks import BaseConfig
from omneval.registry import register_task

def find_metrics(file, metrics):
    assert os.path.exists(file), 'file not found'
    with open(file, 'r') as f:
        text = f.read()
    search_result = re.findall('"'+metrics+'": (.+)?,\n', text)
    if not search_result:
        return -1
    else:
        return float(search_result[0])


@register_task('sst2_test')
class SST2ConfigForUnitTest(BaseConfig):
    task = 'sst2_test'
    task_type = 'classification'
    dataset_name = ['glue', 'sst2']
    metrics = 'accuracy'
    test_subset = 'validation'
    templates = ['sentence|It was |<mask>|.']
    label_mappings = [['terrible'], ['great']]
    labels = [0, 1]
    topk = 1
    eval_batch_size = 32
    max_seq_length = 128



class TestClassification(unittest.TestCase):

    def test_bert(self):
        subprocess.run(['python', 'main.py', 'sst2_test', '--meta_prefix=test'])
        file = 'results/test_sst2.json'
        self.assertAlmostEqual(find_metrics(file, 'accuracy'), .611, delta=1e-2)

