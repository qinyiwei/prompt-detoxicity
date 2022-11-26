import csv
import json
from pathlib import Path
from typing import Optional

import click
from utils.constants import ALLOWED_MODELS

TOPICS = ["computers", "legal", "military",
          "politics", "religion", "science", "space"]


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
def main(output_dir: str, dataset_file: Optional[str], model: str, model_type: str):
    f_prompts = open(dataset_file, 'r')
    srcs = [p.replace("\n", "") for p in f_prompts.readlines()]

    output_dir = Path(output_dir)
    model_name = model.split('/')[-2]+"_"+model.split('/')[-1] \
        if 'checkpoint' in model.split('/')[-1] else model.split('/')[-1]
    log_file = output_dir / f'{model_type}_{model_name}_generations.jsonl'
    wf = open(log_file, 'w')
    writer = csv.DictWriter(
        wf, fieldnames=['category', 'input_text', 'generation'])
    writer.writeheader()

    for topic in TOPICS:
        generations = []
        file_name = "data/topic/generations/gpt2_gpt2-large_n_500_generations_tw_topic_eval_ppl_category_{}_rerank_generations.jsonl".format(
            topic)
        file_name = output_dir / \
            f'{model_type}_{model_name}_{topic}_generations.jsonl'

        with open(file_name, 'r') as f:
            for row in f:
                generations.append(json.loads(row))
        num_gen_per_scr = len(generations)/len(srcs)

        for gen in generations:
            writer.writerow(
                {'category': topic, 'input_text': srcs[int(generations.index(gen)/num_gen_per_scr)], 'generation': gen})


if __name__ == '__main__':
    main()
