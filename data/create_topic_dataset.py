from pathlib import Path
import click

import random

import string
import os
from tqdm import tqdm


def tw_topic_eval(sent, categories, tw_dir, cap=None):
    # num matches of distinct words
    words = {}
    for category in categories:
        with open(os.path.join(tw_dir, category + '.txt'), 'r') as rf:
            word_list = []
            for line in rf:
                word_list.append(line.strip().lower())
            words[category] = word_list

    num_match = [0] * len(categories)
    sent = sent.strip().lower().split()
    sent = [tok.strip(string.punctuation) for tok in sent]
    for id, category in enumerate(categories):
        sent_match = 0
        for word in words[category]:
            if word in sent:
                sent_match += 1
        if cap is not None:
            sent_match += min(cap, sent_match)
        num_match[id] = sent_match
    return num_match


@click.command()
@click.option('--num_topic', default=10000, required=False, type=int, help='')
@click.option('--num_other', default=None, required=False, type=int, help='')
def main(num_topic: int, num_other: int):
    # Create output files
    tw_dir = '/projects/tir4/users/yiweiq/ctrGen/naacl-2021-fudge-controlled-generation/topic_data/wordlists'
    output_dir = '/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/datasets/openwebtext/topic/num_topic_{}_num_other_{}'.format(
        num_topic, num_other)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = Path(output_dir)
    generations_file = '/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/datasets/openwebtext/toxicity_lte2.txt'
    categories = [
        'computers', 'legal', 'military', 'politics', 'religion', 'science', 'space']
    #categories = ['computers']

    f = open(generations_file, 'r')
    generations = [line for line in f.read().splitlines() if (
        len(line) > 0 and not line.isspace())]
    print("start calculating scores:")
    SCORES = []

    for gen in tqdm(generations):
        SCORES.append(tw_topic_eval(gen, categories, tw_dir, cap=None))

    if num_topic is not None:
        for id, category in enumerate(categories):
            sorted_gens = [x for _, x in sorted(
                zip(SCORES, generations), key=lambda pair: pair[0][id], reverse=True)]
            sorted_scores = sorted([s[id] for s in SCORES], reverse=True)
            print("---------------category:{}----------------".format(category))
            print("min_socre orig:{}".format(min(sorted_scores)))
            print("max_socre orig:{}".format(max(sorted_scores)))
            print("avg_socre orig:{}".format(
                sum(sorted_scores)/len(sorted_scores)))
            sorted_scores = sorted_scores[:num_topic]
            print("min_socre:{}".format(min(sorted_scores)))
            print("max_socre:{}".format(max(sorted_scores)))
            print("avg_socre:{}".format(sum(sorted_scores)/len(sorted_scores)))

            output_file = output_dir / \
                '{}.txt'.format(category)
            with open(output_file, 'w') as fo:
                for gen in sorted_gens[:num_topic]:
                    fo.write(gen+'\n')

            print("save {} examples to file:{}".format(num_topic, output_file))

    if num_other is not None:
        sorted_gens = [x for _, x in sorted(
            zip(SCORES, generations), key=lambda pair: sum(pair[0]))]
        sorted_scores = sorted([sum(s) for s in SCORES])
        print("---------------category:{}----------------".format(category))
        print("min_socre orig:{}".format(min(sorted_scores)))
        print("max_socre orig:{}".format(max(sorted_scores)))
        print("avg_socre orig:{}".format(sum(sorted_scores)/len(sorted_scores)))
        sorted_scores = sorted_scores[:num_topic]
        print("min_socre:{}".format(min(sorted_scores)))
        print("max_socre:{}".format(max(sorted_scores)))
        print("avg_socre:{}".format(sum(sorted_scores)/len(sorted_scores)))

        output_file = output_dir / 'other.txt'
        with open(output_file, 'w') as fo:
            for gen in sorted_gens[:num_topic]:
                fo.write(gen+'\n')

        print("save {} examples to file:{}".format(num_other, output_file))


if __name__ == '__main__':
    main()
