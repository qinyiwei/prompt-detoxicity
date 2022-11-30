from typing import Dict, List, Union


import numpy as np
import torch
from transformers import BatchEncoding, GPT2TokenizerFast, T5TokenizerFast
import os
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from transformers.utils import logging
from datasets import load_dataset
import random

logger = logging.get_logger(__name__)


class SingleAttributeDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path_toxicity: str, file_path_safe: str,
                 dataset_name: str = None, mid_th: float = 2):
        self.examples = []

        if dataset_name is not None:
            dataset = load_dataset(dataset_name)
            for split in dataset.keys():
                for example in dataset[split]:
                    if example['label'] > mid_th:
                        attribute = 0
                    else:
                        attribute = 1
                    score = abs(example['label'] - mid_th)
                    self.examples.append(
                        {"text": example['text'], "attribute": attribute, "score": score})
        else:
            assert os.path.isfile(
                file_path_toxicity), f"Input file path {file_path_toxicity} not found"
            assert os.path.isfile(
                file_path_safe), f"Input file path {file_path_safe} not found"
            # Here, we do not cache the features, operating under the assumption
            # that we will soon use fast multithreaded tokenizers from the
            # `tokenizers` repo everywhere =)
            logger.info("Creating features from dataset file at %s",
                        file_path_toxicity)
            logger.info(
                "Creating features from dataset file at %s", file_path_safe)

            # attribute = 0, safe
            self.examples.extend(self._read_file(
                file_path_safe, attribute=0))
            # attribute = 1, toxicity
            self.examples.extend(self._read_file(
                file_path_toxicity, attribute=1))
            print(len(self.examples))

        self.tokenizer = tokenizer

    def _read_file(self, file_path, attribute):
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (
                len(line) > 0 and not line.isspace())]

        examples = [{"text": l, "attribute": attribute, "score": 1}
                    for l in lines]
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class MultiAttributeDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str):
        self.examples = []

        assert os.path.isdir(file_path) or os.path.isfile(
            file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s",
                    file_path)
        logger.info(
            "Creating features from dataset file at %s", file_path)

        if os.path.isdir(file_path):
            self.files = os.listdir(file_path)
            self.files = sorted(self.files)

            for id, file in enumerate(self.files):
                # attribute = 0, safe
                self.examples.extend(self._read_file(
                    os.path.join(file_path, file), attribute=id))
        else:
            self.examples.extend(self._read_file(file_path, attribute=0))
        print(len(self.examples))
        self.tokenizer = tokenizer

    def _read_file(self, file_path, attribute):
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (
                len(line) > 0 and not line.isspace())]

        examples = [{"text": l, "attribute": attribute, "score": 1}
                    for l in lines]
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class SingleAttributeDataCollator:
    def __init__(self, tokenizer, data_args, n_class=2, n_prefix=None):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.n_prefix = n_prefix
        self.n_class = n_class

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        #assert isinstance(examples[0], (dict, BatchEncoding))
        add_tokens = ["<CLS_{}_TOK_{}>".format(str(j), str(i).zfill(2))
                      for j in range(self.n_class) for i in range(self.n_prefix)]

        add_tokens_safe = ["<CLS_{}_TOK_{}>".format(str(0), str(i).zfill(2))
                           for i in range(self.n_prefix)]
        add_tokens_toxicity = ["<CLS_{}_TOK_{}>".format(str(1), str(i).zfill(2))
                               for i in range(self.n_prefix)]

        self.tokenizer.add_tokens(add_tokens)

        token_ids_safe = self.tokenizer(
            "".join(add_tokens_safe), return_tensors="pt")["input_ids"]
        token_ids_toxicity = self.tokenizer(
            "".join(add_tokens_toxicity), return_tensors="pt")["input_ids"]
        if isinstance(self.tokenizer, T5TokenizerFast):
            token_ids_safe = token_ids_safe[:, :-1]
            token_ids_toxicity = token_ids_toxicity[:, :-1]

        assert token_ids_safe.shape[-1] == self.n_prefix
        assert token_ids_toxicity.shape[-1] == self.n_prefix

        batch = {}
        batch["scores"] = torch.tensor(
            [e['score'] for e in examples])

        if not self.data_args.is_T5:
            batch_orig = self._encode_GPT2(examples)
            batch_orig = self.tokenizer.pad(batch_orig, return_tensors="pt")
        else:
            batch_orig = self._encode_T5(examples)

        batch.update(self.input_add_prefix(batch_orig, token_ids_safe, token_ids_toxicity,
                                           name="maxi", is_T5=self.data_args.is_T5))
        batch.update(self.input_add_prefix(batch_orig, token_ids_safe, token_ids_toxicity,
                                           name="mini", is_T5=self.data_args.is_T5))

        return batch

    def input_add_prefix(self, batch_orig, token_ids_safe, token_ids_toxicity, name, is_T5):
        input_ids = batch_orig['input_ids']
        bs = input_ids.shape[0]
        mask_toxi = batch_orig['attribute'].unsqueeze(
            1).repeat(1, token_ids_safe.shape[1])
        mask_safe = 1 - mask_toxi

        if name == "maxi":
            prefixes = torch.mul(mask_toxi, token_ids_toxicity.repeat(
                bs, 1)) + torch.mul(mask_safe, token_ids_safe.repeat(bs, 1))
        if name == "mini":
            prefixes = torch.mul(mask_safe, token_ids_toxicity.repeat(
                bs, 1)) + torch.mul(mask_toxi, token_ids_safe.repeat(bs, 1))

        new_input_ids = torch.cat([prefixes, input_ids], 1)
        new_attention_mask = torch.cat([torch.ones(
            (bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1)

        if not is_T5:
            token_type_ids = torch.cat([torch.zeros(
                (bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1)
            labels = torch.cat(
                [torch.zeros((bs, self.n_prefix), dtype=torch.long), batch_orig["input_ids"]], 1)
            inputs = {
                "input_ids_{}".format(name): new_input_ids,
                "attention_mask_{}".format(name): new_attention_mask,
                "token_type_ids_{}".format(name): token_type_ids,
                "labels_{}".format(name): labels,
            }
        else:
            labels = batch_orig["labels"]
            decoder_input_ids = self._shift_right_t5(labels)
            inputs = {
                "input_ids_{}".format(name): new_input_ids,
                "attention_mask_{}".format(name): new_attention_mask,
                "decoder_input_ids_{}".format(name): decoder_input_ids,
                "labels_{}".format(name): labels,
            }
        return inputs

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode_GPT2(self, batch):
        attributes = [b['attribute'] for b in batch]
        text = [b['text'] for b in batch]

        batch_encoding = self.tokenizer(
            text, add_special_tokens=True, truncation=True, max_length=self.data_args.block_size)
        examples = batch_encoding["input_ids"]
        for e in examples:
            e.append(self.tokenizer.eos_token_id)
        examples = [{"input_ids": torch.tensor(
            e, dtype=torch.long), "attribute": a} for e, a in zip(examples, attributes)]

        return examples

    def _encode_T5(self, batch) -> Dict[str, torch.Tensor]:
        attributes = [b['attribute'] for b in batch]
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["text"][:int(len(x["text"])/2)] for x in batch],
            tgt_texts=[x["text"][int(len(x["text"])/2):] for x in batch],
            max_length=self.data_args.block_size,
            max_target_length=self.data_args.block_size,
            # padding=self.padding,
            return_tensors="pt",
            # **self.dataset_kwargs,
        )
        batch = batch_encoding.data
        batch['attribute'] = torch.tensor(attributes, dtype=torch.long)
        return batch


class AttributeDataCollator:
    def __init__(self, tokenizer, data_args, n_class=2, n_prefix=None):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.n_prefix = n_prefix
        self.n_class = n_class

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode_GPT2(self, batch):
        attributes = [b['attribute'] for b in batch]
        text = [b['text'] for b in batch]

        batch_encoding = self.tokenizer(
            text, add_special_tokens=True, truncation=True, max_length=self.data_args.block_size)
        examples = batch_encoding["input_ids"]
        for e in examples:
            e.append(self.tokenizer.eos_token_id)
        examples = [{"input_ids": torch.tensor(
            e, dtype=torch.long), "attribute": a} for e, a in zip(examples, attributes)]

        return examples

    def _encode_T5(self, batch) -> Dict[str, torch.Tensor]:
        attributes = [b['attribute'] for b in batch]
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["text"][:int(len(x["text"])/2)] for x in batch],
            tgt_texts=[x["text"][int(len(x["text"])/2):] for x in batch],
            max_length=self.data_args.block_size,
            max_target_length=self.data_args.block_size,
            # padding=self.padding,
            return_tensors="pt",
            # **self.dataset_kwargs,
        )
        batch = batch_encoding.data
        batch['attribute'] = torch.tensor(attributes, dtype=torch.long)
        return batch


class MultiAttributeDataCollator(AttributeDataCollator):
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        #assert isinstance(examples[0], (dict, BatchEncoding))
        add_tokens = ["<CLS_{}_TOK_{}>".format(str(j), str(i).zfill(2))
                      for j in range(self.n_class) for i in range(self.n_prefix)]

        add_tokens_dict = {}
        for id in range(self.n_class):
            add_tokens_dict[id] = ["<CLS_{}_TOK_{}>".format(str(id), str(i).zfill(2))
                                   for i in range(self.n_prefix)]

        self.tokenizer.add_tokens(add_tokens)

        for id in range(self.n_class):
            add_tokens_dict[id] = self.tokenizer(
                "".join(add_tokens_dict[id]), return_tensors="pt")["input_ids"]

            if isinstance(self.tokenizer, T5TokenizerFast):
                add_tokens_dict[id] = add_tokens_dict[id][:, :-1]

            assert add_tokens_dict[id].shape[-1] == self.n_prefix

        batch = {}
        batch["scores"] = torch.tensor(
            [e['score'] for e in examples])

        if not self.data_args.is_T5:
            batch_orig = self._encode_GPT2(examples)
            batch_orig = self.tokenizer.pad(batch_orig, return_tensors="pt")
        else:
            batch_orig = self._encode_T5(examples)

        batch.update(self.input_add_prefix(batch_orig, add_tokens_dict,
                                           name="maxi", is_T5=self.data_args.is_T5))
        if len(add_tokens_dict.keys()) > 1:
            batch.update(self.input_add_prefix(batch_orig, add_tokens_dict,
                                               name="mini", is_T5=self.data_args.is_T5))

        return batch

    def input_add_prefix(self, batch_orig, add_tokens_dict, name, is_T5):
        input_ids = batch_orig['input_ids']
        bs = input_ids.shape[0]
        attributes = batch_orig['attribute'].tolist()
        attributes_list = list(add_tokens_dict.keys())

        if name == "maxi":
            prefixes = [add_tokens_dict[attr] for attr in attributes]
            prefixes = torch.cat(prefixes, 0)
        if name == "mini":
            attributes_not = []
            for a in attributes:
                attributes_list_others = attributes_list.copy()
                attributes_list_others.remove(a)
                attributes_not.append(random.choice(attributes_list_others))
            prefixes = [add_tokens_dict[attr] for attr in attributes_not]
            prefixes = torch.cat(prefixes, 0)

        new_input_ids = torch.cat([prefixes, input_ids], 1)
        new_attention_mask = torch.cat([torch.ones(
            (bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1)

        if not is_T5:
            token_type_ids = torch.cat([torch.zeros(
                (bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1)
            labels = torch.cat(
                [torch.zeros((bs, self.n_prefix), dtype=torch.long), batch_orig["input_ids"]], 1)
            inputs = {
                "input_ids_{}".format(name): new_input_ids,
                "attention_mask_{}".format(name): new_attention_mask,
                "token_type_ids_{}".format(name): token_type_ids,
                "labels_{}".format(name): labels,
            }
        else:
            labels = batch_orig["labels"]
            decoder_input_ids = self._shift_right_t5(labels)
            inputs = {
                "input_ids_{}".format(name): new_input_ids,
                "attention_mask_{}".format(name): new_attention_mask,
                "decoder_input_ids_{}".format(name): decoder_input_ids,
                "labels_{}".format(name): labels,
            }
        return inputs
