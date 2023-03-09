# Toolformer-reimplementation

# Introduction

A Python implementation of [Toolformer](https://arxiv.org/abs/2302.04761) using Huggingface Transformers.

The main code of the repo is following the repo [simple-toolformer](https://github.com/mrcabbage972/simple-toolformer).

The main difference are as follow:

- Deepspeed framework

- Improve data generation code following the paper

# TODO

- finetuning code

# Problem

Although setting a relative high threshold($\tau_s=0$), there is still some unsatisfactory case.
```
In one hour, there are 3 sets of 20 minutes.\nSo, Joy can read 8 x 3 = 24 pages in an hour.\nIt will take her [CALCULATOR(24 * 5) -> 120.00] 120/24 = 5 hours to read 120 pages.
```

The generation code is not complete, which can't generate a case with more than one API call.

# Usage
First, please install the requirements file.

The example training script is at `src/scripts/train_gsm8k.py`. This would train the model on the [GSM8k](https://huggingface.co/datasets/gsm8k) dataset of Math Word Problems. 

# Citations

```bibtex
@inproceedings{Schick2023ToolformerLM,
    title   = {Toolformer: Language Models Can Teach Themselves to Use Tools},
    author  = {Timo Schick and Jane Dwivedi-Yu and Roberto Dessi and Roberta Raileanu and Maria Lomeli and Luke Zettlemoyer and Nicola Cancedda and Thomas Scialom},
    year    = {2023}
}
```