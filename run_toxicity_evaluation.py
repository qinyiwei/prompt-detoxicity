from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import click
import pandas as pd
import torch
from tqdm import tqdm
import os
import json
from transformers import pipeline

from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER,ALLOWED_MODELS
from utils.perspective_api import PerspectiveWorker, unpack_scores
from utils.utils import load_jsonl, batchify, ensure_dir,load_cache
import torch.utils.data as data
from training.run_pplm_discrim_train import load_discriminator, collate_fn


def make_generations_col(generations, responses, is_toxicity):
    for generation, response in zip(generations, responses):
        if is_toxicity:
            if response['response']:
                response = unpack_scores(response['response'])[0]
            else:
                response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {'text': generation, **response}


def collate(dataset: Optional[pd.DataFrame], generations: List[str], responses: Iterable[Dict[str, Any]], output_file: str, is_toxicity: bool):
    generations_col_iter = make_generations_col(generations, responses, is_toxicity)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        print(f"Detected samples per prompt:", n)
        generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
        dataset['generations'] = generations_col

    dataset.to_json(output_file, orient='records', lines=True)

class Dataset(data.Dataset):
    def __init__(self, X, discriminator, device, max_length_seq = 128, add_eos_token = True):
        """Reads source and target sequences from txt files."""
        self.X = []
        self.y = []

        for text in X:
            seq = discriminator.tokenizer.encode(text)
            if add_eos_token:
                seq = [discriminator.tokenizer.eos_token_id] + seq
            seq = seq[:max_length_seq]
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            self.X.append(seq)
            self.y.append(0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--perspective-rate-limit', default=25)
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--number', type=int, default=None)
@click.option('--evaluation-method', type=str, default="perspective", help="choose from perspective|toxicity_classifier")
@click.option('--weights_path', required=False, help='')
@click.option('--meta_path', required=False, help='')
@click.option('--batch-size', default=32)
@click.option('--generate-safe', default=True, type=bool, help="generate safe texts or toxicity text")

def main(output_dir: str, dataset_file: Optional[str], model: str, model_type: str, 
    perspective_rate_limit: int, n: int, number: int, evaluation_method: str, weights_path: str, meta_path: str, batch_size: int, generate_safe: bool):
    # Load prompts from dataset file
    assert dataset_file.endswith('.jsonl')
    dataset = pd.read_json(dataset_file, lines=True)
    prompts = pd.json_normalize(dataset['prompt'])['text']

    print('Prompts:', '\n', prompts)

    if number is not None:
        dataset = dataset[:number]
        prompts = prompts[:number]

    # Create output files
    output_dir = Path(output_dir)
    model_name = model.split('/')[-2]+"_"+model.split('/')[-1] \
        if 'checkpoint' in model.split('/')[-1] else model.split('/')[-1]
    generations_file = output_dir / f'{model_type}_{model_name}_{"safe" if generate_safe else "toxicity"}_generations.jsonl'
    toxicity_eval_file = output_dir / f'{model_type}_{model_name}_{"safe" if generate_safe else "toxicity"}_toxicity_eval.jsonl'
    assert os.path.exists(generations_file)   # don't overwrite generations!
    ensure_dir(output_dir)
    output_file = output_dir / f'{"prompted"}_gens_{model_type}_{model_name}_{"safe" if generate_safe else "toxicity"}.jsonl'

    # Generate and collate perspective scores
    generations = []

    for gen in load_cache(generations_file):
        generations.append(gen)
    
    if evaluation_method == "perspective":
        # Create perspective worker thread
        is_toxicity = True
        perspective = PerspectiveWorker(
            out_file=toxicity_eval_file,
            total=len(prompts) * n,
            rate_limit=perspective_rate_limit
        )

        i = 0
        for gen in generations:
            perspective(f'generation-{i}', gen)
            i += 1

        torch.cuda.empty_cache()
        perspective.stop()
        print('Finished generation and perspective scoring!')
    elif evaluation_method == "toxicity_classifier":
        is_toxicity = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        discriminator, meta_param = load_discriminator(weights_path, meta_path, device=device)
        discriminator = discriminator.to(device)
        discriminator.eval()
        eval_dataset = Dataset(generations, discriminator, device, max_length_seq = 128, add_eos_token = True)
        data_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)
        SCORES = []
        with torch.no_grad():
            for input_t,_ in tqdm(data_loader):
                output_t = discriminator(input_t)
                scores = torch.softmax(output_t,1)[:,1]
                SCORES.extend(scores.cpu().detach().numpy().tolist())

        toxicity_f = open(toxicity_eval_file, 'w')
        for i,score in enumerate(SCORES):
            data_example = {
                "request_id": "generation-{}".format(i),
                "response": {"attributeScores":{"TOXICITY":{
                    "spanScores": [{"begin": 0, "end": 0, "score": {"value": score, "type": "PROBABILITY"}}],
                    "summaryScore": {"value": score, "type": "PROBABILITY"}}}
                    }
            } 
            json.dump(data_example, toxicity_f)
            toxicity_f.write("\n")
        toxicity_f.close()
    elif evaluation_method == "sentiment-analysis":
        # score generations and write to sentiment.jsonl
        is_toxicity = False
        classifier = pipeline('sentiment-analysis')
        with open(toxicity_eval_file, 'w') as fo:
            for i, p in tqdm(enumerate(prompts), total=len(prompts), desc='Scoring generations'):
                sentences_for_prompt = []
                for j in range(n):
                    gen = generations[i*n + j]
                    sentences_for_prompt.append(f'{p}{gen}')
                try:
                    predictions_for_prompt = classifier(sentences_for_prompt)
                except IndexError: # sometimes the generation is too long?
                    predictions_for_prompt = [{'label': "", 'score': float('nan')}] * len(sentences_for_prompt)
                for res in predictions_for_prompt:
                    fo.write(json.dumps(res) + '\n')
    else:
        raise NotImplementedError
    
    torch.cuda.empty_cache()
    print('Finished generation and evluation!')

    #Save evaluation file
    if os.path.exists(toxicity_eval_file):
        print('Collating output files')
        collate(dataset, generations, load_jsonl(toxicity_eval_file), output_file, is_toxicity)

    print("save perspective score to {}".format(toxicity_eval_file))
    print("save output file to {}".format(output_file))

if __name__ == '__main__':
    main()