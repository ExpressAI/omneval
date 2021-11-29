import argparse
from omneval.registry import build_config, build_evaluator, build_processor, build_metrics
from omneval.utils import init_eval_result, write_meta_eval_to_json, print_eval_result, check_if_answers_single_tokens
from omneval.config import get_args
import logging
import traceback
import pdb
logging.basicConfig(level=logging.INFO)


def main():
    args = get_args()
    for task in args.tasks.split('|'):
        config = build_config(args, task)
        logging.info(config.__dict__)
        meta_eval = init_eval_result(config)
        logging.info("Working on task: %s"%task)
        if config.task_type == 'classification':
            check_if_answers_single_tokens(config.label_mappings)
        try:
            for arch in args.archs.split('|'):
                config.arch = arch
                logging.info("Working on arch: %s" % arch)
                processor = build_processor(config)
                evaluator = build_evaluator(config)
                metrics_fn = build_metrics(config)
                for pid in range(len(meta_eval['prompts'])):
                    template = processor.prompt_schema(pid)
                    dataset = processor.generate_dataset(pid)
                    aux_input = processor.generate_aux_inputs(pid)
                    output, eval_result = evaluator.eval(dataset, metrics_fn, **aux_input)
                    if config.output_inference:
                        evaluator.write_inference_to_json(output, pid)
                    meta_eval['prompts'][pid]['results'].append(eval_result)
                    print_eval_result(eval_result, template)
        except Exception as e:
            print(arch, 'failed')
            print(traceback.format_exc())

        write_meta_eval_to_json(meta_eval, config)


    ##TODO: add distributed_main
if __name__ == "__main__":
    main()
