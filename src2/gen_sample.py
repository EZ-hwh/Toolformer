import re
import argparse
import torch
import deepspeed
from datasets import load_dataset
from torch.utils.data import DataLoader
from calculator import CalculatorTool
from transformers import AutoModelForCausalLM,AutoTokenizer
from tool import Tool
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

def prepare_dataset_for_sampling(dataset, tokenizer, tool):
    prompts_dataset = dataset.map(lambda x: {'prompt': tool.get_prompt_template().format(x['label']),
                                             'label': x['label']})
    
    encoded_dataset = prompts_dataset.map(lambda x: {"label_ids":tokenizer(x['label'])['input_ids']})
    encoded_dataset = encoded_dataset.map(lambda x: {"input_ids":tokenizer(x['prompt'])['input_ids']+x['label_ids']})
    encoded_dataset = encoded_dataset.map(lambda x: {"attention_mask":len(x['input_ids'])*[1]})
    encoded_dataset = encoded_dataset.map(lambda x: {'target_mask': (len(x['input_ids']) - len(tokenizer(x['label'])['input_ids'])) * [0] + len(tokenizer(x['label'])['input_ids']) * [1]})
    encoded_dataset.set_format(columns=['input_ids', 'attention_mask', 'target_mask'], type='torch')
    print(tokenizer.decode(encoded_dataset[1]['input_ids']))
    return encoded_dataset

class APISampler():
    def __init__(self, model, tokenizer, top_k, num_seq_per_pos, tau_s=5e-2,tau_f=1):
        self.model = model
        self.tokenizer = tokenizer
        self.num_seq_per_pos = num_seq_per_pos
        self.top_k = top_k
        self.tau_s = tau_s
        self.tau_f = tau_f
        self.tool_call_token_id = tokenizer.convert_tokens_to_ids(Tool.API_CALL_PREFIX)
        self.tool_call_end_token_id = tokenizer.convert_tokens_to_ids(Tool.API_CALL_SUFFIX)
    def gen_sample(self,dataset:Dataset,tool:Tool):
        self.dataset=prepare_dataset_for_sampling(dataset,self.tokenizer,tool)
        self.tool=tool
        print(self.dataset.__len__())
        #first, generate api position
        self.__gen_topk_pos()
        print(self.dataset.__len__())
        self.dataset.to_json("sample_topk.jsonl")
        #second, generate api call
        self.__gen_call()
        print(self.dataset.__len__())
        self.dataset.to_json("sample_api.jsonl")
        #third, filter
        self.__filter()
        print(self.dataset.__len__())
        self.dataset.to_json("sample_api_filter.jsonl")
    def __gen_topk_pos(self):
        data_loader = DataLoader(self.dataset)
        all_pred_ids = []
        all_pred_probs = []
        for inputs in tqdm(data_loader, desc='Sampling topk positions'):
            with torch.no_grad():
                model_inputs = {k: v.to('cuda') for k, v in inputs.items() if k != 'target_mask'}
                out = self.model(**model_inputs)
                api_prob_at_idx = torch.softmax(out.logits, dim=-1)[:, :, self.tool_call_token_id] # 采样所有生成的token中，[的概率
                api_prob_at_idx[inputs['target_mask'] == 0] = -10000 # 将padding的位置的概率设为-10000, FIXME: 写的有问题
                api_topk = api_prob_at_idx.topk(self.top_k) # 采样TopK的位置
                api_topk_idx, api_topk_prob = api_topk.indices, api_topk.values
                all_pred_ids.append(api_topk_idx.detach().cpu())
                all_pred_probs.append(api_topk_prob.detach().cpu())
                torch.cuda.empty_cache()
        a,b=torch.cat(all_pred_ids, 0), torch.cat(all_pred_probs, 0)
        dataset=Dataset.from_dict({col: [] for col in self.dataset.column_names})
        for i in range(self.top_k):
            tmp=self.dataset.map(lambda x,idx:{'pos':a[idx,i].item(),'prob':b[idx,i].item(),'n':a[idx,i].item()-len(x['input_ids'])+len(x['label_ids'])},with_indices=True).remove_columns('target_mask')
            tmp=tmp.filter(lambda x:x['prob']>self.tau_s)
            dataset=concatenate_datasets([dataset,tmp])
        self.dataset=dataset
    def __gen_call(self):
        self.dataset=self.dataset.map(lambda x:{'input_ids':torch.cat([x['input_ids'][:x['pos']+1],torch.tensor([self.tool_call_token_id])]),'attention_mask':torch.cat([x['attention_mask'][:x['pos']+1],torch.tensor([1])])})
        data_loader = DataLoader(self.dataset)
        api_list=[]
        for inputs in tqdm(data_loader,desc='Generate API'):
            with torch.no_grad():
                inputs = {"input_ids": inputs['input_ids'].to('cuda'),"attention_mask":inputs['attention_mask'].to('cuda')}
                batch_output=self.model.generate(**inputs,
                                    max_new_tokens=50,
                                    num_return_sequences=self.num_seq_per_pos,
                                    do_sample=True,
                                    eos_token_id=self.tool_call_end_token_id,
                                    pad_token_id=self.tool_call_end_token_id)
                api_list.append(batch_output)
        dataset=Dataset.from_dict({col: [] for col in self.dataset.column_names})
        for i in range(self.num_seq_per_pos):
            tmp=self.dataset.map(lambda x,idx:{"api":self.tokenizer.decode(api_list[idx][i][x['pos']+2:],skip_special_tokens=True)},with_indices=True)
            tmp=tmp.filter(lambda x:x['api'][-1]==']')
            tmp=tmp.map(lambda x:{'api':x['api'].replace(']','')})
            dataset=concatenate_datasets([dataset,tmp])
        dataset=dataset.map(lambda x:{'answer':self.tool.run(x['api'])})
        dataset=dataset.filter(lambda x:len(x['answer'])!=0)
        self.dataset=dataset
    def __filter(self):
        self.dataset=self.dataset.map(lambda x:{'P':' [' +x['api']+' -> '+x['answer']+']','N':' [' +x['api']+' -> ]'})
        #calculate L()
        self.dataset=self.dataset.map(lambda x:{"input_ids":x['label_ids']})
        self.__cal_loss('L')
        #calculate L(z)
        self.dataset=self.dataset.map(lambda x:{"input_ids":self.tokenizer(x['N'])['input_ids']+list(x['label_ids'])})
        self.__cal_loss('Lz')
        #calculate L(z,ans)
        self.dataset=self.dataset.map(lambda x:{"input_ids":self.tokenizer(x['P'])['input_ids']+list(x['label_ids'])})
        self.__cal_loss('Lza')
        #filter
        self.dataset.to_json("sample_api_before_filter.jsonl")
        self.dataset=self.dataset.filter(lambda x:x['Lza']<min(x['L'],x['Lz'])-self.tau_f)
    def __cal_loss(self,name):
        dataset=self.dataset.map(lambda x:{"attention_mask":(len(x['input_ids']))*[1],"pos":x["n"]+len(x['input_ids'])-len(x['label_ids'])})
        dataset.set_format(columns=['input_ids','attention_mask','pos'], type='torch')
        data_loader=DataLoader(dataset)
        loss_list=[]
        for inputs in tqdm(data_loader,desc="Cal "+name):
            with torch.no_grad():
                model_inputs = {"input_ids": inputs['input_ids'].to('cuda'),"attention_mask":inputs['attention_mask'].to('cuda')}
                out = self.model(**model_inputs)
                p = torch.softmax(out.logits, dim=-1)[0]
                p=p.detach().cpu()
                w=1
                l=0
                pos=inputs['pos'].item()
                sum_w=0
                for i in range(pos,min(pos+5,p.shape[0]-1)):
                    l-=w*torch.log(p[i][inputs['input_ids'][0][i+1]]).item()
                    sum_w+=w
                    w-=0.2
                loss_list.append(0 if sum_w==0 else l/sum_w)
        self.dataset=self.dataset.add_column(name,loss_list)

                
def main():
    parser = argparse.ArgumentParser(description="Toolformer")
    parser.add_argument("--local_rank", dest="local_rank", type=int, default=-1, help="local rank")
    parser=deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    torch.manual_seed(42)
    args.device = torch.device('cuda')
    model_name="EleutherAI/gpt-j-6B"
    model=AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    engine=deepspeed.init_inference(model)
    model=engine.module
    model.eval()
    data_files = {'train': '../dataset/train.jsonl', 'test': '../dataset/test.jsonl'}
    dataset = load_dataset('json', data_files=data_files, split='train').select(range(2500))
    dataset = dataset.rename_column('question', 'input')
    dataset = dataset.rename_column('answer', 'label')
    dataset = dataset.map(lambda x: {'input': x['input'],
                                     'label': re.sub("(<<).*?(>>)", "", x['label']).split('####')[0].rstrip().replace('\n','')})
    
    api_sampler = APISampler(model,tokenizer,4,8)
    api_sampler.gen_sample(dataset,CalculatorTool())

if __name__ == '__main__':
    main()
