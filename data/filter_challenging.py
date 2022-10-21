orig_file = '/projects/tir4/users/yiweiq/toxicity/dataset/realtoxicityprompts-data/prompts.jsonl'
output_file = '/projects/tir4/users/yiweiq/toxicity/dataset/realtoxicityprompts-data/prompts_challenging.jsonl'


def load_prompts(filename: str, challenging_only: bool = False) -> List[Prompt]:
    """Load prompts from a jsonl file in the RealToxicityPrompts format"""
    print(f'Loading prompts from file "{filename}"')
    prompts = []
    with open(filename, 'r', encoding='utf8') as fh:
        for line in fh:
            line_json = json.loads(line)
            prompt = Prompt(text=line_json['prompt']['text'], challenging=line_json['challenging'])
            if not challenging_only or prompt.challenging:
                prompts.append(prompt)
    print(f'Done loading {len(prompts)} {"challenging " if challenging_only else ""}prompts from file "{filename}"')
    return prompts