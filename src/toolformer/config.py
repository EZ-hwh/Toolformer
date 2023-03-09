from dataclasses import dataclass

import torch


@dataclass
class ToolformerConfig:
    # General
    model_name = "EleutherAI/gpt-j-6B"
    #model_name = "EleutherAI/gpt-neo-1.3B"
    #model_name = "EleutherAI/gpt-neo-2.7B"
    #model_name = "gpt2"
    causal_model = True

    # Sampling
    sampler = 'two_step'

    # Inference
    max_new_tokens = 128

    # Training
    mlm_prob = 0.15
    max_length = 256
    output_path = '..'
    output_name = 'model'
    learning_rate = 1e-5
    train_batch_size = 16
    eval_batch_size = 1
    epochs = 1
    weight_decay = 0.01
    warmup_ratio = 0.1
    fp16 = True
    early_stopping_patience = 1
    test_size = 0.2

    # Filtering
    tool_call_thresh = 0.5

    # File paths
    tool_call_samples_path = '../tool_dataset/{}_for_{}.jsonl'
    tool_call_samples_path_filtered = '../tool_dataset/{}_for_{}.jsonl'