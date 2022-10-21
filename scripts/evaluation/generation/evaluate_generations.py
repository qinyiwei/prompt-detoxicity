"""
evaluate generated output for diversity (dist-n) and fluency (perplexity according to GPT2-XL)
"""

import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

'''
def conditional_perplexity(generations_df, model, tokenizer, device='cuda'):
    perplexities = []
    ct = 0
    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating fluency'):
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
        # for every generation conditioned on the prompt
        generations = [g['text'] for g in row['generations']]
        for gen in generations:
            full_input_ids = tokenizer.encode(prompt+gen, return_tensors='pt').to(device)
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
            loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
            ppl = math.exp(loss.item())
            if ppl < 1e4:   # for sanity
                perplexities.append(ppl)
    return np.nanmean(perplexities)

'''
def get_loss(model, tokenizer, device, texts):
    loss_fct = torch.nn.NLLLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    lsm = torch.nn.LogSoftmax(dim=1)

    def query_model_batch_gen(model, tokenizer, device, texts):
        inputs = tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}
        output = model(**inputs,labels=inputs['input_ids'])
        #print("lm_loss:{}".format(output['loss']))
        return inputs, output['logits']

    inputs, logits = query_model_batch_gen(model, tokenizer, device, texts)
    tgt_tokens = inputs['input_ids']
    logits = logits[..., :-1, :].contiguous()
    tgt_tokens = tgt_tokens[..., 1:].contiguous()
    tgt_len = inputs['attention_mask'].sum(dim=1)-1

    logits = logits.view(-1, model.config.vocab_size)
    loss = loss_fct(lsm(logits), tgt_tokens.view(-1))
    loss = loss.view(tgt_tokens.shape[0], -1)
    #prob = -loss.sum(dim=1) 
    #prob = prob/tgt_len
    loss = loss.sum(dim=1) 

    return loss,tgt_len

def conditional_perplexity(generations_df, model, tokenizer, device='cuda'):
    perplexities = []
    ct = 0
    # for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating fluency'):
        prompt = row.prompt['text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
        # for every generation conditioned on the prompt
        generations = [g['text'] for g in row['generations']]
        batch_gen = [prompt+gen for gen in generations]

        full_loss, tgt_len = get_loss(model, tokenizer, device, batch_gen) 
        losses = (full_loss - prompt_loss) / (tgt_len - (prompt_input_ids.shape[1]-1))
        
        ppl = [math.exp(loss.item()) for loss in losses]
        #if ppl < 1e4:   # for sanity
        perplexities.extend([p for p in ppl if (p < 1e4 and p != 0)])
    return np.nanmean(perplexities)


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating diversity'):
        generations = [g['text'] for g in row['generations']]
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)
    
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


@click.command()
@click.option('--output-dir', required=False)
@click.option('--model', required=False, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=False, type=str)
@click.option('--generations_file', required=False, type=str, help='a jsonl file with generations and attribute scores')
@click.option('--generate-safe', default=True, type=bool, help="generate safe texts or toxicity text")
@click.option('--ppl-model', default="gpt2-xl", type=str, help="generate safe texts or toxicity text")

def main(output_dir: str, model:str, model_type:str, generations_file:str, generate_safe:bool, ppl_model:str):
    
    # Create output files
    if generations_file is None:
        output_dir = Path(output_dir)
        model_name = model.split('/')[-2]+"_"+model.split('/')[-1] \
            if 'checkpoint' in model.split('/')[-1] else model.split('/')[-1]
        generations_file = output_dir / f'{"prompted"}_gens_{model_type}_{model_name}_{"safe" if generate_safe else "toxicity"}.jsonl'
    
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    generations_df = pd.read_json(generations_file, lines=True)

    # calculate diversity
    dist1, dist2, dist3 = distinctness(generations_df)
    print("dist1-{},dist2-{},dist3-{}".format(dist1,dist2,dist3))

    # write output results
    with open(output_dir / 'eval_results.txt', 'w') as fo:
        for i, dist_n in enumerate([dist1, dist2, dist3]):
            fo.write(f'dist-{i+1} = {dist_n}\n')

    # calculate fluency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model = AutoModelForCausalLM.from_pretrained(ppl_model).to(device)
    eval_tokenizer = AutoTokenizer.from_pretrained(ppl_model)
    if "gpt2" in ppl_model:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

    torch.cuda.empty_cache()
    with torch.no_grad():
        ppl = conditional_perplexity(generations_df, eval_model, eval_tokenizer, device=device)
    print("ppl:{}".format(ppl))

    # write output results
    with open(output_dir / 'eval_results.txt', 'a') as fo:
        fo.write(f'perplexity = {ppl}')
    print("write output to {}/eval_results.txt".format(output_dir))

if __name__ == '__main__':
    main()