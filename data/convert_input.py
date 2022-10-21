
import json
from tqdm import tqdm
import random 

#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/gpt2/prompted_gens_gpt2.jsonl"
#output_file = open("convert_generations/prompted_gens_gpt2.jsonl",'w')

#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/dapt/prompted_gens_gpt2.jsonl"
#output_file = open("convert_generations/dapt_prompted_gens_gpt2.jsonl",'w')

#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/dexperts/large_experts/prompted_gens_dexperts.jsonl"
#output_file = open("convert_generations/large_prompted_gens_dexperts.jsonl",'w')

#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/gedi/prompted_gens_gedi.jsonl"
#output_file = open("convert_generations/prompted_gens_gedi.jsonl",'w')

orig_input_file = "/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/pplm/prompted_gens_pplm.jsonl"
output_file = open("convert_generations/prompted_gens_pplm.jsonl",'w')

#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/dataset/realtoxicityprompts-data/generations/prompted/prompted_gens_gpt2.jsonl"
#output_file = open("convert_generations/rtp_prompted_gens_gpt2.jsonl",'w')
#output_file_part = open("convert_generations/rtp_part_prompted_gens_gpt2.jsonl",'w')

examples = []
#orig_data = json.load(open(orig_input_file,'r'))
for line in tqdm(open(orig_input_file,'r')):
    #data.append(json.loads(line))
    one_prompt = json.loads(line)
    for generation in one_prompt["generations"]:
        text = generation["text"]
        profanity = generation["profanity"]
        flirtation = generation["flirtation"]
        sexually_explicit = generation["sexually_explicit"]
        insult = generation["insult"]
        identity_attack = generation["identity_attack"]
        severe_toxicity = generation["severe_toxicity"]
        toxicity = generation["toxicity"]
        threat = generation["threat"]

        if toxicity is None:
            continue
        output_data = {
            "text":text,
            "profanity":profanity,
            "flirtation":flirtation,
            "sexually_explicit":sexually_explicit,
            "insult":insult,
            "identity_attack":identity_attack,
            "severe_toxicity":severe_toxicity,
            "toxicity":toxicity,
            "threat":threat,
        }
        json.dump(output_data, output_file)
        output_file.write("\n")
        examples.append(output_data)

'''
dev_examples_per_class = 1000 
test_examples_per_class = 10000 

attribute_name = 'toxicity'
examples = sorted(examples, key=lambda example: example[attribute_name])

total_examples_per_class = dev_examples_per_class + test_examples_per_class
bottom_examples = examples[:total_examples_per_class]
top_examples = examples[-total_examples_per_class:]

sum_bottom = sum(ex[attribute_name] for ex in bottom_examples) / total_examples_per_class
print(f'Average "{attribute_name}" for bottom examples: '
        f'{sum_bottom}')
sum_top = sum(ex[attribute_name] for ex in top_examples) / total_examples_per_class
print(f'Average "{attribute_name}" for top examples: '
        f'{sum_top}')

rng = random.Random(42)
rng.shuffle(bottom_examples)
rng.shuffle(top_examples)

all_dev_examples = bottom_examples[:dev_examples_per_class] + top_examples[:dev_examples_per_class]
all_test_examples = bottom_examples[dev_examples_per_class:] + top_examples[dev_examples_per_class:]

examples = all_dev_examples + all_test_examples
for example in examples:
    json.dump(example, output_file_part)
    output_file_part.write("\n")
'''