import torch
import argparse
from tasks import build_config, build_evaluator, build_processor
import logging
import pdb
import collections
logging.basicConfig(level=logging.INFO)



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
    for pid in range(processor.prompt_count):
        dataset = processor.generate_dataset(pid)
        aux_input = processor.generate_aux_inputs(pid)
        outputs.append(evaluator.eval(dataset, **aux_input))
        if args.out_dir:
            evaluator.write_to_json(outputs, args.out_dir)

    # logging.info("%s of task %s, prompt %d: %.3f" % (config.metrics, args.task, pid, evaluator.eval(dataset, **aux_input)))


##TODO: add distributed_main
if __name__ == "__main__":
    main()
