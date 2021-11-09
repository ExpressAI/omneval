import argparse
from omneval.registry import build_config, build_evaluator, build_processor
from omneval.utils import init_eval_result, write_meta_eval_to_json, print_eval_result
import logging
logging.basicConfig(level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser(description='user settings for AutoEval')
    parser.add_argument("tasks", type=str, default='sst2')
    parser.add_argument("--archs", type=str, default='bert-base-uncased')
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--calibrate", action='store_true')
    parser.add_argument("--out_dir", type=str, default='results')
    parser.add_argument("--out_format", type=str, default='json')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    for task in args.tasks.split('|'):
        config = build_config(args, task)
        meta_eval = init_eval_result(config)
        logging.info("Working on task: %s"%task)
        for arch in args.archs.split('|'):
            config.arch = arch
            logging.info("Working on arch: %s" % arch)
            processor = build_processor(config)
            evaluator = build_evaluator(config)
            for pid in range(len(meta_eval['prompts'])):
                template = processor.prompt_schema(pid)
                dataset = processor.generate_dataset(pid)
                aux_input = processor.generate_aux_inputs(pid)
                output, eval_result = evaluator.eval(dataset, **aux_input)
                if config.out_dir:
                    evaluator.write_inference_to_json(output, pid)
                meta_eval['prompts'][pid]['results'].append(eval_result)
                print_eval_result(eval_result, template)
        write_meta_eval_to_json(meta_eval, config)


    ##TODO: add distributed_main
if __name__ == "__main__":
    main()
