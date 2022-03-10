import argparse


def get_args():
    """Experiment parameters"""
    parser = argparse.ArgumentParser(description='user settings for AutoEval')
    parser.add_argument("tasks", type=str, default='sst2',
                        help="define the evaluation astk, task configuration should be initialized in the task_zoo")
    parser.add_argument("--archs", type=str, default='bert-base-uncased',
                        help="define PLMs for evaluation, multiple PLMs separated by '|' ")
    parser.add_argument("--max_seq_length", type=int,
                        help="Maximum length for input(include prompt template")
    parser.add_argument("--calibrate", action='store_true',
                        help='Whether to use calibration method')
    parser.add_argument("--out_dir", type=str, default='results',
                        help="The directory for ")
    parser.add_argument("--meta_prefix", type=str, default='meta')
    parser.add_argument("--output_inference", action='store_true')
    args = parser.parse_args()
    return args