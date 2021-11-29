import collections
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='user settings for AutoEval')
parser.add_argument("task", type=str, default='sst2')
parser.add_argument("--out_dir", type=str, default='results')
parser.add_argument("--metrics", type=str, default='accuracy')
args = parser.parse_args()

with open(args.out_dir+'/meta_%s.json'%args.task, 'r') as f:
    df = json.load(f)
memo = collections.defaultdict(list)
for item in df['prompts']:
    for result in item['results']:
        memo[result['plm']].append(result[args.metrics][-1])

for k, v in memo.items():
    print(k, args.metrics, ':', np.mean(v), 'std:', np.std(v), 'cnt: ', len(v))