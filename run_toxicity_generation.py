from pathlib import Path
from typing import Optional

import click
import pandas as pd
import os

from generation.generation import gpt2, gpt2_prompt, gpt2_debias
from utils.constants import ALLOWED_MODELS
from utils.utils import ensure_dir

PATTERNS_generative_LM = {
    'toxicity': 'The following text contains rude, disrespectful, unreasonable or unappropriate language:',
    'safe': 'The following text contains polite, respectful, reasonable and appropriate language:',
}

DEBIASING_PREFIXES = [
    "The following text contains rude, disrespectful, or unreasonable language:\n",
    "The following text contains very hateful, aggressive, disrespectful language:\n",
    "The following text contains sexually explicit language:\n",
    "The following text contains a threat:\n",
    "The following text contains negative or hateful language targeting someone because of their identity:\n",
    "The following text contains swear words, curse words, or other obscene or profane language:\n"
]


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--orig_model', default=None, required=False)
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=32)
@click.option('--top_p', default=0.9, type=float,
              help='Only for sampling. If set, only the top_p tokens with the highest probabilities are considered.')
@click.option('--top_k', default=0, type=int,
              help='Only for sampling. If set, only the top_k tokens with the highest probabilities are considered.')
@click.option('--num_beams', default=1, type=int,
              help='The number of beams for beam search')
@click.option('--do_sample', default=True, type=bool, help="Whether to perform sampling instead of greedy decoding.")
@click.option('--decay_constant', default=50, type=int,
              help='Decay constant for self debias method.')
@click.option('--tuning_type', type=str, default=None)
@click.option('--n_prefix', type=int, default=20)
@click.option('--n_class', type=int, default=2)
@click.option('--number', type=int, default=None)
@click.option('--generate-safe', default=True, type=bool, help="generate safe texts or toxicity text")
@click.option('--use-debias-prefixes-all', default=False, type=bool, help="whether to use all 6 debias prefixes")
def main(output_dir: str, dataset_file: Optional[str], model: str, orig_model: str, model_type: str, n: int, max_tokens: int, batch_size: int,
         top_p: float, top_k: int, num_beams: int, do_sample: bool, decay_constant: int, tuning_type: str,
         n_prefix: int, n_class: int, number: int, generate_safe: bool, use_debias_prefixes_all: bool):
    # Load prompts from dataset file
    if dataset_file.endswith('.jsonl'):
        dataset = pd.read_json(dataset_file, lines=True)
        prompts = pd.json_normalize(dataset['prompt'])['text']
    else:
        f_prompts = open(dataset_file, 'r')
        # 1
        prompts = f_prompts.readlines()
        # 2
        #prompts = [p[7:] for p in prompts]
        # 3
        #prompts = [p[7:] for p in prompts if p.startswith('[ WP ]')]

        prompts = pd.Series(prompts)

    print('Prompts:', '\n', prompts)

    if number is not None:
        dataset = dataset[:number]
        prompts = prompts[:number]

    # Create output files
    output_dir = Path(output_dir)
    model_name = model.split('/')[-2]+"_"+model.split('/')[-1] \
        if 'checkpoint' in model.split('/')[-1] else model.split('/')[-1]
    generations_file = output_dir / \
        f'{model_type}_{model_name}_{"safe" if generate_safe else "toxicity"}_generations.jsonl'
    print("generation file:{}".format(generations_file))
    # don't overwrite generations!
    assert not os.path.exists(generations_file)
    ensure_dir(output_dir)

    # Setup model for generation
    # TODO: move this logic into generation.py
    if model_type == 'gpt2':
        generations_iter = gpt2(
            prompts=prompts,
            add_params=None,
            max_len=max_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            top_k=top_k,
            num_samples=n,
            top_p=top_p,
            batch_size=batch_size,
            model_name_or_path=model,
            orig_model=orig_model,
            out_file=generations_file
        )
    elif model_type.startswith("gpt2_prompt"):
        if model_type == 'gpt2_prompt_fixed':
            add_params = PATTERNS_generative_LM['safe'] if generate_safe else PATTERNS_generative_LM['toxicity']
        elif model_type == 'gpt2_prompt_tunable':
            assert tuning_type == "prompt_tuning"
            add_tokens_safe = ["<CLS_{}_TOK_{}>".format(str(0), str(i).zfill(2))
                               for i in range(n_prefix)]
            add_tokens_toxicity = ["<CLS_{}_TOK_{}>".format(str(1), str(i).zfill(2))
                                   for i in range(n_prefix)]
            add_params = "".join(
                add_tokens_safe if generate_safe else add_tokens_toxicity)
        else:
            raise NotImplementedError
        generations_iter = gpt2_prompt(
            prompts=prompts,
            add_params=add_params,
            max_len=max_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            top_k=top_k,
            num_samples=n,
            top_p=top_p,
            batch_size=batch_size,
            model_name_or_path=model,
            orig_model=orig_model,
            out_file=generations_file,
            tuning_type=tuning_type,
            n_prefix=n_prefix,
            n_class=n_class,
        )
    elif model_type == 'gpt2_debias':
        if tuning_type == None:
            debiasing_prefixes_neg = DEBIASING_PREFIXES if use_debias_prefixes_all else DEBIASING_PREFIXES[
                0:1]
            debiasing_prefixes_pos = None
        elif tuning_type == 'prompt_tuning':
            debiasing_prefixes_safe = ["<CLS_{}_TOK_{}>".format(str(0), str(i).zfill(2))
                                       for i in range(n_prefix)]
            debiasing_prefixes_toxicity = ["<CLS_{}_TOK_{}>".format(str(1), str(i).zfill(2))
                                           for i in range(n_prefix)]
            if generate_safe:
                debiasing_prefixes_neg = ["".join(debiasing_prefixes_toxicity)]
                debiasing_prefixes_pos = ["".join(debiasing_prefixes_safe)]
            else:
                debiasing_prefixes_neg = ["".join(debiasing_prefixes_safe)]
                debiasing_prefixes_pos = ["".join(debiasing_prefixes_toxicity)]
        else:
            raise NotImplementedError
        generations_iter = gpt2_debias(
            prompts=prompts,
            max_len=max_tokens,
            min_length=None,
            do_sample=do_sample,
            num_beams=num_beams,
            top_k=top_k,
            num_samples=n,
            batch_size=batch_size,
            model_name_or_path=model,
            orig_model=orig_model,
            out_file=generations_file,
            top_p=top_p,
            debiasing_prefixes=debiasing_prefixes_neg,
            decay_constant=decay_constant,
            epsilon=0.01,
            debug=False,
            num_return_sequences=1,
            is_debias=True,
            tuning_type=tuning_type,
            n_prefix=n_prefix,
            n_class=n_class,
        )

    else:
        raise NotImplementedError(f'Model {model} not implemented')

    # Generate
    generations = []
    for i, gen in enumerate(generations_iter):
        generations.append(gen)

    print("save file to {}".format(generations_file))


if __name__ == '__main__':
    main()
