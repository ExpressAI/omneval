import torch
import argparse
from tasks import build_config, build_evaluator, build_processor
import logging
import pdb
import collections
logging.basicConfig(level=logging.INFO)
import numpy as np



def get_args():
    parser = argparse.ArgumentParser(description='user settings for AutoEval')
    parser.add_argument("task", type=str, default='sst2')
    parser.add_argument("--arch", type=str, default='bert-base-uncased')
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--calibrate", action='store_true')
    parser.add_argument("--out_dir", type=str, default='testx3.json')
    args = parser.parse_args()
    return args


def main():
    #TODO: Build a universal get_args functions for all config for tasks or evluation processes
    args = get_args()
    config = build_config(args)
    processor = build_processor(config)
    evaluator = build_evaluator(config)
    outputs = []
    eval_results = {}
    for pid in range(processor.prompt_count):
        dataset = processor.generate_dataset(pid)
        aux_input = processor.generate_aux_inputs(pid)
        output, eval_result = evaluator.eval(dataset, **aux_input)
        outputs.append(output)
        eval_results[processor.prompt_schema(pid)] = eval_result
    if args.out_dir:
        evaluator.write_to_json(outputs, args.out_dir)
    logging.info(eval_results)
    metrics_avg = np.mean([v[config.metrics] for v in eval_results.values()])
    logging.info("Average Evaluation metrics of the task %s on model: %s---%s: %.3f" % (
       config.task, config.arch, config.metrics, metrics_avg))

        ##TODO: add distributed_main
if __name__ == "__main__":
    main()
