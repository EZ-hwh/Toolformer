import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler
from transformers import DataCollatorWithPadding
import torch.nn.functional as F
from toolformer.tool import Tool
from tqdm import tqdm

def prepare_dataset_for_sampling(dataset, tokenizer, tool):
    prompts_dataset = dataset.map(lambda x: {'prompt': tool.get_prompt_template().format(x['label']) + x['label'],
                                             'label': x['label']})
    encoded_dataset = prompts_dataset.map(lambda x: tokenizer(x['prompt'],truncation=True, padding=True),batched=True)
    encoded_dataset = encoded_dataset.map(lambda x: {'target_mask': (len(x['input_ids']) - len(tokenizer(x['label'])['input_ids'])) * [0] + len(tokenizer(x['label'])['input_ids']) * [1]})
    encoded_dataset.set_format(columns=['input_ids', 'attention_mask', 'target_mask'], type='torch')
    #print(tokenizer.decode(encoded_dataset[0]['input_ids']))
    return encoded_dataset

def postprocess_samples(all_preds, dataset, tool, is_causal_model, duplicate_samples):
    pred_ds = Dataset.from_dict({'text': all_preds,
                                 'prompt': [tool.get_prompt_template().format(z['label']) for z in dataset for _ in range(duplicate_samples)]})
    if is_causal_model:
        return pred_ds.map(lambda x: {'text': x['text'][len(x['prompt']):].split(']')[0] + ']'})
        #return pred_ds.map(lambda x: {'text': x['text'][x['text'].rindex('Output: ') + len('Output: '):].split(']')[0] + ']'})
    else:
        return pred_ds.map(lambda x: {'text': x['text'].split(']')[0] + ']'})

class BasicToolSampler:
    """
    A basic tool sampler that just calls the generate method of them model
    """
    def __init__(self, tokenizer, model, cfg ,args):
        self.cfg = cfg
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def sample(self, dataset: Dataset, tool: Tool) -> Dataset:
        self.model.eval()
        encoded_dataset = prepare_dataset_for_sampling(dataset, self.tokenizer, tool)

        train_sampler = DistributedSampler(encoded_dataset) if self.args.local_rank != -1 else RandomSampler(encoded_dataset)
        params = {"batch_size": self.cfg.eval_batch_size, "sampler": train_sampler}
        data_loader = DataLoader(encoded_dataset,
                   collate_fn=DataCollatorWithPadding(self.tokenizer), **params)
        data_iter = iter(data_loader)

        all_preds = []
        with torch.no_grad():
            for inputs in data_iter:
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                batch_preds = self.model.generate(**inputs,
                                                  max_new_tokens=self.cfg.max_new_tokens,
                                                  return_dict_in_generate=True,
                                                  output_scores=True)
                                                  #repetition_penalty=1.0,
                                                  #top_p=0.5)
                all_preds += [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch_preds['sequences']]
                print(all_preds[-1])
                print('#'*100)
                torch.cuda.empty_cache()
                #inputs = {k: v.to('cpu') for k, v in inputs.items()}

        # This is a bit ugly due to iterating over the dataset manually
        return postprocess_samples(all_preds, dataset, tool, self.cfg.causal_model)


class TwoStepToolSampler:
    """
    WORK IN PROGRESS
    Implements the sampling procedure as detailed in the paper:
     First, sample K positions for the [ token.
     Then, sample M sequences out of each of the K.
    """

    def __init__(self, tokenizer, model, cfg, args, top_k, num_seq_per_pos, tau_s=0.05):
        self.cfg = cfg
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.num_seq_per_pos = num_seq_per_pos
        self.top_k = top_k
        self.tau_s = tau_s
        self.tool_call_token_id = tokenizer.convert_tokens_to_ids(Tool.API_CALL_PREFIX)
        #print(self.tool_call_token_id)
        self.tool_call_end_token_id = tokenizer.convert_tokens_to_ids(Tool.API_CALL_SUFFIX)

    def sample(self, dataset: Dataset, tool: Tool) -> Dataset: 
        encoded_dataset = prepare_dataset_for_sampling(dataset, self.tokenizer, tool)

        topk_pos_idx, topk_pos_prob = self.get_topk_pos_idx(encoded_dataset, tool) # 生成TopK最高概率生成[的位置
        anns_at_pos = [self.get_anns_at_pos(dataset, encoded_dataset, topk_pos_idx[:, idx], topk_pos_prob[:, idx], tool) for idx in range(self.top_k)]
        return concatenate_datasets(anns_at_pos)

    def get_topk_pos_idx(self, encoded_dataset, tool):
        data_loader = DataLoader(encoded_dataset, batch_size=self.cfg.eval_batch_size,
                                 collate_fn=DataCollatorWithPadding(self.tokenizer))
        data_iter = iter(data_loader)

        # TODO: create mask to cancel all tokens from the prompt template

        all_pred_ids = []
        all_pred_probs = []
        for inputs in tqdm(data_iter, desc='Sampling topk positions'):
            with torch.no_grad():
                model_inputs = {k: v.to(self.args.device) for k, v in inputs.items() if k != 'target_mask'}
                out = self.model(**model_inputs)
                # FIXME: 添加了一个Softmax层，变成按照生成概率进行采样
                #api_prob_at_idx = out.logits[:, :, self.tool_call_token_id] # 采样所有生成的token中，[的概率
                api_prob_at_idx = torch.softmax(out.logits, dim=-1)[:, :, self.tool_call_token_id] # 采样所有生成的token中，[的概率
                api_prob_at_idx[inputs['target_mask'] == 0] = -10000 # 将padding的位置的概率设为-10000, FIXME: 写的有问题
                api_topk = api_prob_at_idx.topk(self.top_k) # 采样TopK的位置
                api_topk_idx, api_topk_prob = api_topk.indices, api_topk.values
                all_pred_ids.append(api_topk_idx.detach())
                all_pred_probs.append(api_topk_prob.detach())
                torch.cuda.empty_cache()
        return torch.cat(all_pred_ids, 0), torch.cat(all_pred_probs, 0)

    def get_anns_at_pos(self, dataset, encoded_dataset, pos_idx, pos_prob, tool):
        # TODO: refactor to avoid having to pass two dataset objects
        # Get the text before the desired position and add the tool call token
        dataset_filter = dataset.add_column('pos_prob', pos_prob.cpu().numpy())\
            .filter(lambda x: x['pos_prob'] > self.tau_s)
        dataset_at_idx = encoded_dataset.add_column('pos_idx', pos_idx.cpu().numpy())\
            .add_column('pos_prob', pos_prob.cpu().numpy())\
            .filter(lambda x: x['pos_prob'] > self.tau_s)\
            .map(lambda x: {'input_ids': torch.cat([x['input_ids'][:x['pos_idx'] + 1], torch.tensor(self.tool_call_token_id).unsqueeze(-1)], -1),
                            'text': self.tokenizer.decode(x['input_ids'][:x['pos_idx'] + 1], skip_special_tokens=True)
                                    + self.tokenizer.decode(torch.tensor(self.tool_call_token_id).unsqueeze(-1), skip_special_tokens=True),
                            'attention_mask': torch.cat([x['attention_mask'][:x['pos_idx'] + 1], torch.tensor(1).unsqueeze(-1)], -1),
                            'suffix': self.tokenizer.decode(x['input_ids'][x['pos_idx'] + 1:], skip_special_tokens=True)})

        #print(dataset_at_idx[0])
        suffixes_ds = dataset_at_idx.map(lambda x: {'suffix': x['suffix']})
        nsuffixes_ds = Dataset.from_dict({'suffix': [i['suffix'] for i in suffixes_ds for _ in range(self.num_seq_per_pos)]})

        dataset_at_idx.set_format(columns=['input_ids', 'attention_mask'], type='torch')
        data_loader = DataLoader(dataset_at_idx, batch_size=self.cfg.eval_batch_size,
                                 collate_fn=DataCollatorWithPadding(self.tokenizer))
        data_iter = iter(data_loader)

        all_preds = []
        for inputs in tqdm(data_iter, 'Generate API'):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_preds = self.model.generate(**inputs,
                                                  max_new_tokens=self.cfg.max_new_tokens,
                                                  return_dict_in_generate=True,
                                                  output_scores=True,
                                                  num_return_sequences=self.num_seq_per_pos,
                                                  do_sample=True,
                                                  eos_token_id=self.tool_call_end_token_id,
                                                  pad_token_id=self.tool_call_end_token_id)
                torch.cuda.empty_cache()
            all_preds += [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch_preds['sequences']]
        samples_ds = postprocess_samples(all_preds, dataset_filter, tool, self.cfg.causal_model, self.num_seq_per_pos)
        samples_ds = concatenate_datasets([samples_ds, nsuffixes_ds], axis=1)

        return samples_ds.map(lambda x: {'text': x['text'] + x['suffix']})