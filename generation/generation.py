# TODO: add `text` key to cached generations
# TODO: consolidate code for loading cache
import json
import logging
import math
from functools import partial
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers.pipelines import pipeline

from generation.gpt2_generation import GPT2Generation
from generation.gpt2_add_generation import GPT2AddGeneration
from modeling.modeling import GPT2Wrapper

from utils.utils import batchify, load_cache



logging.disable(logging.CRITICAL)  # Disable logging from transformers

def _pipeline_helper(prompts: pd.Series,
                     model_name_or_path: str,
                     max_len: int,
                     num_samples: int,
                     out_file: Path,
                     p: float,
                     **generate_kwargs):
    # Load cached generations
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1
    assert num_cached_generations % num_samples == 0

    # Remove prompts that have already been generated with
    prompts = prompts[num_cached_generations // num_samples:]
    if prompts.empty:
        return

    # Setup model
    generator = pipeline('text-generation', model=model_name_or_path, device=0)
    print("Created pipeline with model:", generator.model.__class__.__name__)

    # Generate with prompts
    for prompt in tqdm(prompts, desc='Generation', dynamic_ncols=True):
        # Generate
        # FIXME: this is a hack
        ctx_len = len(generator.tokenizer.tokenize(prompt))
        try:
            batch = generator(prompt,
                              num_return_sequences=num_samples,
                              clean_up_tokenization_spaces=True,
                              do_sample=True,
                              top_p=p,
                              max_length=ctx_len + max_len,
                              return_prompt=False,
                              **generate_kwargs)
            batch = map(lambda g: g['generated_text'][len(prompt):], batch)
        except RuntimeError as e:
            print("Error during generation with prompt:", prompt)
            print(e)
            print("Emptying CUDA cache and continuing...")
            torch.cuda.empty_cache()

            batch = ["GENERATION_ERROR_CUDA"] * num_samples

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def ctrl(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         ctrl_code: str,
         model_name_or_path: str,
         out_file: Path,
         **generate_kwargs) -> Iterable[str]:
    # Prepend CTRL code to prompts
    prompts = ctrl_code + " " + prompts
    print(prompts)

    yield from _pipeline_helper(prompts=prompts,
                                model_name_or_path=model_name_or_path,
                                max_len=max_len,
                                num_samples=num_samples,
                                out_file=out_file,
                                **generate_kwargs)


def _gpt2_helper(prompts: pd.Series,
                 max_len: int,
                 num_samples: int,
                 batch_size: int,
                 generator: GPT2Generation,
                 out_file: Path,
                 **generate_kwargs):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    for prompt in tqdm(batchify(prompts, batch_size),
                       total=math.ceil(len(prompts) / batch_size),
                       desc=f'Generation',
                       dynamic_ncols=True,
                       postfix={'batch_size': batch_size}):
        # Generate
        batch = generator.generate(prompt, max_len, **generate_kwargs)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def gpt2(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         batch_size: int,
         model_name_or_path: str,
         out_file: Path,
         **generate_kwargs) -> Iterable[str]:
    # Setup model
    generator = GPT2Generation(model_name_or_path)

    yield from _gpt2_helper(prompts=prompts,
                            max_len=max_len,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            generator=generator,
                            out_file=out_file,
                            **generate_kwargs)


def _gpt2_add_helper(prompts: pd.Series,
                 add_params: str,
                 max_len: int,
                 num_samples: int,
                 batch_size: int,
                 generator: GPT2Generation,
                 out_file: Path,
                 **generate_kwargs):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    for prompt in tqdm(batchify(prompts, batch_size),
                       total=math.ceil(len(prompts) / batch_size),
                       desc=f'Generation',
                       dynamic_ncols=True,
                       postfix={'batch_size': batch_size}):
        # Generate
        batch = generator.generate(prompt, add_params, max_len, **generate_kwargs)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def gpt2_add(prompts: pd.Series,
         add_params: str,
         max_len: int,
         num_samples: int,
         batch_size: int,
         model_name_or_path: str,
         out_file: Path,
         tuning_type: str,
         n_prefix: int,
         n_class: int,
         **generate_kwargs) -> Iterable[str]:
    # Setup model
    generator = GPT2AddGeneration(
        model_name_or_path = model_name_or_path,
        tokenizer_name_or_path = model_name_or_path,
        tuning_type = tuning_type, 
        n_prefix = n_prefix, 
        n_class = n_class,
    )

    yield from _gpt2_add_helper(prompts=prompts,
                            add_params=add_params,
                            max_len=max_len,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            generator=generator,
                            out_file=out_file,
                            **generate_kwargs)


def _gpt2_debias_helper(prompts: pd.Series,
                 max_len: int,
                 num_samples: int,
                 batch_size: int,
                 generator: GPT2Wrapper,
                 out_file: Path,
                 **generate_kwargs):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    for prompt in tqdm(batchify(prompts, batch_size),
                       total=math.ceil(len(prompts) / batch_size),
                       desc=f'Generation',
                       dynamic_ncols=True,
                       postfix={'batch_size': batch_size}):
        # Generate
        batch = generator.generate_self_debiasing(prompt, max_length = max_len, **generate_kwargs)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def gpt2_debias(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         batch_size: int,
         model_name_or_path: str,
         out_file: Path,
         is_debias: bool,
         **generate_kwargs) -> Iterable[str]:
    # Setup model
    generator = GPT2Wrapper(model_name=model_name_or_path, is_debias=is_debias)

    yield from _gpt2_debias_helper(prompts=prompts,
                            max_len=max_len,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            generator=generator,
                            out_file=out_file,
                            **generate_kwargs)

