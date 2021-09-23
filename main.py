import torch
import argparse
from tasks import build_config, build_evaluator, build_processor
import logging
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description='user settings for AutoEval')
    parser.add_argument("task", type=str, default='sst2')
    parser.add_argument("--arch", type=str, default='bert-base-uncased')
    parser.add_argument("--max_seq_length", type=int, default=128)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    task_config = build_config(args.task)
    arch = args.arch
    processor = build_processor(arch, task_config)
    evaluator = build_evaluator(arch, task_config)
    for pid in range(processor.prompt_count):
        dataset = processor.generate_dataset(pid)
        aux_input = processor.generate_aux_inputs(pid)
        logging.info("Accuracy of task %s, prompt %d: %.2f" % (args.task, pid, evaluator.eval(dataset, **aux_input)))


##TODO: add distributed_main
if __name__ == "__main__":
    main()
