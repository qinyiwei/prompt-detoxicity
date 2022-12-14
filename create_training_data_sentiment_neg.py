from pathlib import Path
import torch
from tqdm import tqdm
import click

from utils.utils import load_cache
import torch.utils.data as data
from training.run_pplm_discrim_train import load_discriminator, collate_fn
import random


class Dataset(data.Dataset):
    def __init__(self, X, discriminator, device, max_length_seq=1024, add_eos_token=True):
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


def save_part_file(SCORES, generations, output_dir, rate=None, num=None):
    sorted_gens = [x for _, x in sorted(
        zip(SCORES, generations), reverse=True)]

    # Save evaluation file
    if rate is not None and num is None:
        output_file = output_dir / \
            'negativity_gte99_most_neg_rate_{}.txt'.format(rate)
        with open(output_file, 'w') as fo:
            for gen in sorted_gens[:int(len(generations)*rate/100)]:
                fo.write(gen+'\n')
    elif num is not None and rate is None:
        output_file = output_dir / \
            'negativity_gte99_most_neg_num_{}.txt'.format(num)
        with open(output_file, 'w') as fo:
            for gen in sorted_gens[:num]:
                fo.write(gen+'\n')
    elif num is not None and rate is not None:
        output_file = output_dir / \
            'negativity_gte99_most_neg_rate_{}_num_{}.txt'.format(rate, num)
        filtered_gens = sorted_gens[:int(len(generations)*rate/100)]
        random.seed(0)
        mini_indices = random.sample(
            list(range(len(filtered_gens))), min(num, len(filtered_gens)))
        filtered_gens = [filtered_gens[index] for index in mini_indices]
        with open(output_file, 'w') as fo:
            for gen in filtered_gens[:num]:
                fo.write(gen+'\n')
    print("rate is:")
    print(rate)
    print("num is:")
    print(num)
    print("save negative texts to {}".format(output_file))


@click.command()
@click.option('--rates', required=False, multiple=True, type=int, help='')
@click.option('--nums', required=False, multiple=True, type=int, help='')
def main(rates: int, nums: int):
    weights_path = "/projects/tir4/users/yiweiq/toxicity/prompt-detoxicity/models/pplm_classifiers/sentiment_classifierhead_1280/SST_classifier_head_epoch_10.pt"
    meta_path = "/projects/tir4/users/yiweiq/toxicity/prompt-detoxicity/models/pplm_classifiers/sentiment_classifierhead_1280/SST_classifier_head_meta.json"

    # Create output files
    output_dir = '/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/datasets/openwebtext/'
    output_dir = Path(output_dir)
    batch_size = 128

    generations_file = output_dir / 'negativity_gte99_num_all.txt'

    f = open(generations_file, 'r')
    generations = [line for line in f.read().splitlines() if (
        len(line) > 0 and not line.isspace())]

    print("start loading dataset:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator, meta_param = load_discriminator(
        weights_path, meta_path, device=device)
    discriminator = discriminator.to(device)
    discriminator.eval()
    eval_dataset = Dataset(generations, discriminator,
                           device, max_length_seq=128, add_eos_token=True)
    data_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)
    print("start calculating scores:")
    SCORES = []
    with torch.no_grad():
        for input_t, _ in tqdm(data_loader):
            output_t = discriminator(input_t)
            scores = torch.softmax(output_t, 1)[:, 1]
            SCORES.extend(scores.cpu().detach().numpy().tolist())

    print("number of scores:")
    print(len(SCORES))
    print(min(SCORES))
    print(max(SCORES))
    print("number of generations:")
    print(len(generations))
    torch.cuda.empty_cache()
    print('Finished generation and evluation!')

    if len(rates) != 0 and len(nums) == 0:
        for rate in rates:
            save_part_file(SCORES, generations, output_dir, rate=rate)
    elif len(nums) != 0 and len(rates) == 0:
        for num in nums:
            save_part_file(SCORES, generations, output_dir, num=num)
    elif len(nums) != 0 and len(rates) != 0:
        for num in nums:
            for rate in rates:
                save_part_file(SCORES, generations,
                               output_dir, num=num, rate=rate)


if __name__ == '__main__':
    main()
