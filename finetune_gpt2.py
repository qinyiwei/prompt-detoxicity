# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from modeling.modeling_add import set_extra_embeddings, freeze_LM
from training.trainer import ContrastiveTrainer
from utils.data import SingleAttributeDataCollator, SingleAttributeLineByLineTextDataset
import torch

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    add_tuning: Optional[str] = field(
        default=None, metadata={"help": "which addiongal tuning strategies to use, possible choices: prompt_tuning"}
    )
    freeze_LM: bool = field(
        default=True, metadata={"help": "Freeze parameters of LM."}
    )
    n_prefix: int = field(
        default=20,
        metadata={"help": "The number of prefix tokens."},
    )
    is_T5: bool = field(
        default=False,
        metadata={"help": "Whether our model is T5."},
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "dropout rate"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_file_toxicity: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_file_safe: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    n_class: int = field(
        default=2,
        metadata={"help": "The number of classes."},
    )
    margin: Optional[float] = field(
        default=1, metadata={"help": "brio loss margin"}
    ) 
    loss_beta: Optional[float] = field(
        default=1, metadata={"help": ""}
    ) 
    loss_alpha: Optional[float] = field(
        default=1, metadata={"help": ""}
    )       
    loss_type: Optional[int] = field(
        default=1, metadata={"help": "which addiongal tuning strategies to use, possible choices: 1|2"}
    )
    dataset_type: Optional[str] = field(
        default='text', metadata={"help": "which addiongal tuning strategies to use, possible choices: text|single_attribute"}
    )
    learning_rate_LM: Optional[float] = field(
        default=5e-5, metadata={"help": "learning rate for language model."}
    ) 

def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    if args.line_by_line:
        if args.dataset_type == 'text':
            return LineByLineTextDataset(tokenizer=tokenizer, 
                file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
        elif args.dataset_type == 'single_attribute':
            return SingleAttributeLineByLineTextDataset(tokenizer=tokenizer, 
                file_path_toxicity=args.train_data_file_toxicity, 
                file_path_safe=args.train_data_file_safe)
        else:
            raise NotImplementedError
    else:
        return TextDataset(
            tokenizer=tokenizer, 
            file_path=args.eval_data_file if evaluate else args.train_data_file, 
            block_size=args.block_size, overwrite_cache=args.overwrite_cache
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.margin = data_args.margin
    training_args.loss_type = data_args.loss_type
    training_args.loss_alpha = data_args.loss_alpha
    training_args.loss_beta = data_args.loss_beta
    training_args.learning_rate_LM = data_args.learning_rate_LM
    training_args.freeze_LM = model_args.freeze_LM
    training_args.is_T5 = model_args.is_T5
    data_args.is_T5 = model_args.is_T5

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    setattr(config,'dropout_rate',model_args.dropout_rate)

    if model_args.tokenizer_name:
        if model_args.model_type == 'gpt2':
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, pad_token="<|endoftext|>", cache_dir=model_args.cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        if model_args.model_type == 'gpt2':
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, pad_token="<|endoftext|>", cache_dir=model_args.cache_dir)
            #assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id
            #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    if model_args.freeze_LM:
        freeze_LM(model)
    #model.resize_token_embeddings(len(tokenizer))

    if model_args.add_tuning == 'prompt_tuning':
        set_extra_embeddings(model, model_args.n_prefix, data_args.n_class, is_T5 = model_args.is_T5)

    if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        state_dict = torch.load(model_args.model_name_or_path+"/pytorch_model.bin")
        model.transformer.wte.embed._load_from_state_dict(
                {"weight": state_dict["transformer.wte.embed.weight"]}, "", None, True, [], [], "")
        model.transformer.wte.new_embed._load_from_state_dict(
                {"weight": state_dict["transformer.wte.new_embed.weight"]}, "", None, True, [], [], "")


    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    if model_args.add_tuning is None:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
        )
    elif model_args.add_tuning == 'prompt_tuning':
        data_collator = SingleAttributeDataCollator(
            tokenizer=tokenizer, data_args = data_args, n_class = data_args.n_class, n_prefix = model_args.n_prefix
        )
    else:
        raise NotImplementedError

    # Initialize our Trainer
    if model_args.add_tuning is None:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
    elif model_args.add_tuning == 'prompt_tuning':
        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
    else:
        raise NotImplementedError
        

    # Training
    if training_args.do_train:
        #model_path = (
        #    model_args.model_name_or_path
        #    if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
        #    else None
        #)
        #trainer.train(model_path=model_path)
        trainer.train()

        trainer.save_model()

        if trainer.is_world_process_zero():
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()