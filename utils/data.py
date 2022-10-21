from typing import Dict, List, Union


import numpy as np
import torch
from transformers import BatchEncoding
import os
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class SingleAttributeLineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path_toxicity: str, file_path_safe: str, block_size: int):
        assert os.path.isfile(file_path_toxicity), f"Input file path {file_path_toxicity} not found"
        assert os.path.isfile(file_path_safe), f"Input file path {file_path_safe} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path_toxicity)
        logger.info("Creating features from dataset file at %s", file_path_safe)

        self.tokenizer = tokenizer
        self.block_size = block_size
        
        self.examples = []
        self.examples.extend(self._read_file(file_path_safe, attribute = 0)) # attribute = 0, safe
        self.examples.extend(self._read_file(file_path_toxicity, attribute = 1)) # attribute = 1, toxicity
        print(len(self.examples))

    def _read_file(self, file_path, attribute):
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)
        examples = batch_encoding["input_ids"]
        for e in examples:
            e.append(self.tokenizer.eos_token_id)
        examples = [{"input_ids": torch.tensor(e, dtype=torch.long), "attribute": attribute} for e in examples]

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

class SingleAttributeDataCollator:
    def __init__(self, tokenizer, n_class = 2, n_prefix = None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.n_prefix = n_prefix
        self.n_class = n_class

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        assert isinstance(examples[0], (dict, BatchEncoding))
        batch_orig = self.tokenizer.pad(examples, return_tensors="pt")

        add_tokens = ["<CLS_{}_TOK_{}>".format(str(j), str(i).zfill(2)) 
            for j in range(self.n_class) for i in range(self.n_prefix)]

        add_tokens_safe = ["<CLS_{}_TOK_{}>".format(str(0), str(i).zfill(2)) 
            for i in range(self.n_prefix)]
        add_tokens_toxicity = ["<CLS_{}_TOK_{}>".format(str(1), str(i).zfill(2)) 
            for i in range(self.n_prefix)]

        self.tokenizer.add_tokens(add_tokens)

        token_ids_safe = self.tokenizer("".join(add_tokens_safe), return_tensors="pt")["input_ids"]
        token_ids_toxicity = self.tokenizer("".join(add_tokens_toxicity), return_tensors="pt")["input_ids"]
        
        assert token_ids_safe.shape[-1] == self.n_prefix
        assert token_ids_toxicity.shape[-1] == self.n_prefix

        batch = {}
        batch.update(self.input_add_prefix(batch_orig, token_ids_safe, token_ids_toxicity, name = "maxi"))
        batch.update(self.input_add_prefix(batch_orig, token_ids_safe, token_ids_toxicity, name = "mini"))

        return batch

    def input_add_prefix(self, batch_orig, token_ids_safe, token_ids_toxicity, name):
        input_ids = batch_orig['input_ids']
        bs = input_ids.shape[0]
        mask_toxi = batch_orig['attribute'].unsqueeze(1).repeat(1,token_ids_safe.shape[1])
        mask_safe = 1 - mask_toxi
        
        if name == "maxi":
            prefixes = torch.mul(mask_toxi, token_ids_toxicity.repeat(bs, 1)) + torch.mul(mask_safe, token_ids_safe.repeat(bs, 1)) 
        if name == "mini":
            prefixes = torch.mul(mask_safe, token_ids_toxicity.repeat(bs, 1)) + torch.mul(mask_toxi, token_ids_safe.repeat(bs, 1)) 

        new_input_ids = torch.cat([prefixes, input_ids], 1)
        new_attention_mask = torch.cat([torch.ones((bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1)
        token_type_ids = torch.cat([torch.zeros((bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1)
        labels=torch.cat([torch.zeros((bs, self.n_prefix), dtype=torch.long), batch_orig["input_ids"]], 1)

        inputs = {
            "input_ids_{}".format(name): new_input_ids,
            "attention_mask_{}".format(name): new_attention_mask,
            "token_type_ids_{}".format(name): token_type_ids,
            "labels_{}".format(name): labels,
        }

        return inputs
