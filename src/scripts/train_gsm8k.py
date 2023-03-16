import logging
import re, json
import sys
import argparse
import torch
import deepspeed

from datasets import load_dataset

sys.path.append('..')
#print(sys.path)
from toolformer.toolformer import Toolformer
from toolformer.tools.calculator import CalculatorTool
from toolformer.tools.date import DateTool
from transformers.deepspeed import HfDeepSpeedConfig

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Toolformer")
    #parser.add_argument("--cuda", dest="cuda", type=str,default="0,1,2,3,4,5,6,7", help="gpu id")
    parser.add_argument("--ds_config_path", dest="ds_config_path", type=str,default='./ds_config.json', help="deepspeed config json")
    parser.add_argument("--local_rank", dest="local_rank", type=int, default=-1, help="local rank")
    args = parser.parse_args()

    with open(args.ds_config_path) as f:
        ds_config = json.load(f)

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # torch.distributed.init_process_group(backend="nccl")
        deepspeed.init_distributed()
    args.device = device
    dschf = HfDeepSpeedConfig(ds_config)

    tf = Toolformer(args, ds_config)

    data_files = {'train': '../../dataset/train.jsonl', 'test': '../../dataset/test.jsonl'}
    dataset = load_dataset('json', data_files=data_files, split='train').select(range(2))
    #dataset = load_dataset("gsm8k", 'main', split="train").select(range(2))
    dataset = dataset.rename_column('question', 'input')
    dataset = dataset.rename_column('answer', 'label') # TODO: remove math annotations from answers
    dataset = dataset.map(lambda x: {'input': x['input'],
                                     'label': re.sub("(<<).*?(>>)", "", x['label']).split('####')[0].rstrip()})
    
    
    apis = [CalculatorTool()]

    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()
    tf.fit(dataset, apis)


if __name__ == '__main__':
    main()