import argparse





# TODO: need to find ways to combine task config and args
def get_args():
    parser = argparse.ArgumentParser(description='user settings for AutoEval')
    parser.add_argument("tasks", type=str, default='sst2')
    parser.add_argument("--archs", type=str, default='bert-base-uncased')
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--calibrate", action='store_true')
    parser.add_argument("--out_dir", type=str, default='results')
    # parser.add_argument("--out_format", type=str, default='json')
    parser.add_argument("--meta_prefix", type=str, default='meta')
    parser.add_argument("--output_inference", action='store_true')
    args = parser.parse_args()
    return args