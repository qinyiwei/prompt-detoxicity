
import json
from tqdm import tqdm
import numpy as np
import click
from pathlib import Path


@click.command()
@click.option('--output-dir', required=False, type=str,
              help='')
@click.option('--model', required=False, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=False, type=str)
@click.option('--generations_file', required=False, default=None, type=str, help='a jsonl file with generations and attribute scores')
@click.option('--generate-safe', default=True, type=bool, help="generate safe texts or toxicity text")

def main(output_dir:str, model:str, model_type:str, generations_file: str, generate_safe: bool):
    # Get generations_file
    if generations_file is None:
        output_dir = Path(output_dir)
        model_name = model.split('/')[-2]+"_"+model.split('/')[-1] \
            if 'checkpoint' in model.split('/')[-1] else model.split('/')[-1]
        generations_file = output_dir / f'{"prompted"}_gens_{model_type}_{model_name}_{"safe" if generate_safe else "toxicity"}.jsonl'

    POS = []
    for line in tqdm(open(generations_file,'r')):
        one_prompt = json.loads(line)
        label_pos = []
        for generation in one_prompt["generations"]:
            if generation["label"] == 'POSITIVE':
                label_pos.append(1)
            elif generation["label"] == 'NEGATIVE':
                label_pos.append(0)
        POS.append(sum(label_pos)/len(label_pos))

    avg_pos = sum(POS)/len(POS)
    print("POS rate:{}, neg rate:{}".format(avg_pos, 1-avg_pos))


if __name__ == '__main__':
    main()