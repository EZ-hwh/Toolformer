import logging
import os, json
from typing import List

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, T5ForConditionalGeneration, AutoTokenizer, \
    AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling

from toolformer.config import ToolformerConfig
from toolformer.sequence_scoring import get_scores_for_labels
from toolformer.tool import Tool
from toolformer.tool_sampling import BasicToolSampler, TwoStepToolSampler

import deepspeed

logger = logging.getLogger(__name__)

class Toolformer:
    def __init__(self, args, ds_config):
        self.cfg = ToolformerConfig()
        self.args = args
        self.ds_config = ds_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, padding_side='left', pad_token_id = 50256)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.cfg.causal_model:
            self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name)
        else: # Currently assuming that only non-causal model is T5 family
            self.model = T5ForConditionalGeneration.from_pretrained(self.cfg.model_name)

        self.model = deepspeed.init_inference(self.model)
        #self.model,_,_,_=deepspeed.initialize(model=self.model,config_params=ds_config,optimizer=None,lr_scheduler=None)
        self.model.eval()
        if self.cfg.sampler == 'basic':
            self.tool_sampler = BasicToolSampler(self.tokenizer, self.model, self.cfg, args)
        elif self.cfg.sampler == 'two_step':
            # TODO: remove hard-coded args
            self.tool_sampler = TwoStepToolSampler(self.tokenizer, self.model, self.cfg, args, 4, 4)
        else:
            raise ValueError

    def save_samples(self, samples, tool, sample_type):
        #samples.save_to_disk(self.cfg.tool_call_samples_path.format(sample_type, tool.get_tool_name()))
        with open(self.cfg.tool_call_samples_path.format(sample_type, tool.get_tool_name()), 'w') as f:
            for data in samples:
                f.write(json.dumps(data) + '\n')

    def fit(self, dataset: Dataset, tools: List[Tool]):
        """
            This is the main method for implementing the training process described in the Toolformer paper.
            It contains three stages:
            1. Use the model to generate samples of tool usage, using few-shot prompting
            2. Filter the samples by a measure of how much the improve the likelihood of the text after the tool call.
            3. Fit the model on the filtered samples.
        :param dataset:
            The dataset to fit on.
        :param tools:
            A list of tools to be considered.
        """
        logger.info('Fitting with a dataset of size {} and {} tools'.format(len(dataset), len(tools)))
        samples_for_tuning = []
        for tool in tools:
            maybe_tool_samples = self.sample_dataset(dataset, tool)
            logger.info('Examples of {} tool generation results: \n{}'.format(tool.get_tool_name(), '\n\n'.join(maybe_tool_samples[:10]['text'])))
            logger.info('#'*100)
            tool_samples = maybe_tool_samples.filter(lambda x: tool.text_has_call(x['text']))
            logger.info('{} samples left after filtering for tool name'.format(len(tool_samples)))
            if len(tool_samples) > 0:
                logger.info('Examples of {} tool filtered annotations: \n{}'.format(tool.get_tool_name(), '\n\n'.join(tool_samples[:10]['text'])))
                logger.info('#'*100)
                self.save_samples(tool_samples, tool, 'tool_call_samples')
                # FIXME: 返回的格式不对，需要和execute_tool_call的返回格式一致
                valid_tool_samples = tool_samples.filter(lambda x: self.filter_tool_call(x, tool))
                executed_tool_samples = valid_tool_samples.map(lambda x: self.execute_tool_call(x, tool))
                likely_samples = self.filter_likelihood(executed_tool_samples, tool)
                logger.info('{} samples left after filtering by likelihood'.format(len(likely_samples)))
                logger.info('#'*100)
                if len(likely_samples) > 0:
                    self.save_samples(likely_samples, tool, 'tool_call_samples_likely')
                    samples_for_tuning.append(likely_samples)
                    logger.info('Examples of {} after filtering by likelihood: \n{}'.format(tool.get_tool_name(), '\n\n'.join(likely_samples[:5]['text'])))
                    logger.info('#'*100)
        if len(samples_for_tuning) > 0:
            dataset_for_tuning = concatenate_datasets(samples_for_tuning)
        else:
            dataset_for_tuning = []
        if len(dataset_for_tuning) == 0:
            raise ValueError("Can't proceed: There is no data to fine-tune on!")
        #self.fine_tune(dataset_for_tuning)

    def sample_dataset(self, dataset: Dataset, tool: Tool) -> Dataset:
        """
            This methods samples a dataset to produce example API calls.
            The sampling procedure is implemented as just straightforward generation.
            The method in the paper is to first find the top K positions for the next token being [ and then
            to try generating M calls start from each of these K.
        :param dataset:
            The input texts
        :param tool:
            The tool to annotate the input texts with
        :return:
            A Dataset containing a text field and a score field.
        """
        logger.info('Sampling dataset')
        return self.tool_sampler.sample(dataset, tool)


    def filter_likelihood(self, inputs: Dataset, tool: Tool) -> Dataset:
        """
            Filters the sampled tool uses by a criterion that quantifies how much they improve the likelihood
            of the text after the tool call. The paper uses a weighting scheme which is currently not implemented here.
            Another thing to note is that in this stage the tool annotation is prepended to the input text rather than
            inserted at its correct place
            The criterion can be roughly described as:
            # loss (with prefix of tool call and result ) < min(loss (with prefix of tool call), loss(no tool call)
        :param inputs:
            A dataset with tool call annotations.
        :param tool:
            THe tool.
        :return:
            Same as inputs but filtered by the criterion.
        """
        # TODO: implement weighting scheme

        logger.info('Filtering generated samples by their likelihood')

        #if self.cfg.causal_model:
        #    raise NotImplementedError
        
        inputs = inputs.map(lambda x: {**x,
                                       'text_before': tool.get_text_before_call(x['text']),
                                       'tool_call': tool.get_call_from_text(x['text']),
                                       'text_after': tool.get_text_after_call(x['text'])})

        inputs = inputs.map(lambda x: {**x,
                                       'tool_call_wo_result': x['tool_call'][:-1] + ' ' + Tool.RESULT_PREFIX + Tool.API_CALL_SUFFIX,
                                       'tool_call_with_result': x['tool_call'][:-1] + ' ' + Tool.RESULT_PREFIX + ' ' + x['tool_result'] + Tool.API_CALL_SUFFIX,
                                       })

        inputs = inputs.map(lambda x: {**x,
                                       'tool_call_text_before': x['tool_call_wo_result'] + ' ' + x['text_before'],
                                       'tool_call_result_text_before': x['tool_call_with_result'] + ' ' + x['text_before'],
                                       'text': x['text_before'] + x['tool_call_with_result'] + x['text_after']
                                       })

        inputs = inputs.map(lambda x: {**x,
                                       'loss_no_tool':
                                           get_scores_for_labels(x['text_before'], x['text_after'], self.model,
                                                                 self.tokenizer, self.cfg.causal_model, self.args),
                                       'loss_tool':
                                           get_scores_for_labels(x['tool_call_result_text_before'], x['text_after'],
                                                                 self.model, self.tokenizer, self.cfg.causal_model, self.args),
                                       'loss_tool_no_result':
                                           get_scores_for_labels(x['tool_call_text_before'], x['text_after'],
                                                                 self.model, self.tokenizer, self.cfg.causal_model, self.args)
                                       })
        #inputs.set_format(type='torch', columns=['text','text_before','loss_no_tool', 'loss_tool', 'loss_tool_no_result'])

        return inputs.filter(
            lambda x: min(x['loss_no_tool'], x['loss_tool_no_result']) - x['loss_tool'] >= self.cfg.tool_call_thresh)

    def fine_tune(self, api_call_samples: Dataset):
        """
            This is just standard HF fine-tuning with the language modeling objective.
            See e.g. https://huggingface.co/docs/transformers/tasks/language_modeling
        :param api_call_samples:
        """

        self.model, self.optimizer, _, _  = deepspeed.initialize(model=self.model, config_params=ds_config, model_parameters=self.model.parameters())

        logger.info('Fine-tuning the model on {} API call samples'.format(len(api_call_samples)))
        datasets = api_call_samples.train_test_split(test_size=self.cfg.test_size)

        train_args = TrainingArguments(
            output_dir=os.path.join(self.cfg.output_path, self.cfg.output_name),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=self.cfg.learning_rate,
            per_device_train_batch_size=self.cfg.train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            num_train_epochs=self.cfg.epochs,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            load_best_model_at_end=True,
            save_total_limit=1,
            #deepspeed='ds_config.json',
            fp16=self.cfg.fp16,
        )

        if self.cfg.causal_model:
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.cfg.mlm_prob)

        trainer = Trainer(
            self.model,
            train_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            tokenizer=self.tokenizer,
            compute_metrics=None,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.cfg.early_stopping_patience)]
        )

        trainer.train()

    def filter_tool_call(self, sample, tool: Tool) -> bool:
        return tool.run(sample['text']) != ''
    
    def execute_tool_call(self, sample, tool: Tool) -> dict:
        sample['tool_result'] = tool.run(sample['text'])
        return sample