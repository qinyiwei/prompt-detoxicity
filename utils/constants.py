from pathlib import Path
import yaml

# Config
CONFIG_FILE = Path('config.yml')
PERSPECTIVE_API_KEY = ''
OPENAI_API_KEY = ''
try:
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    PERSPECTIVE_API_KEY = config['perspective']
    OPENAI_API_KEY = config['openai']
except FileNotFoundError:
    print('No config file found. API keys will not be loaded.')

# Perspective API

PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
ALLOWED_MODELS = ['gpt2', 'gpt2_prompt_fixed', 'gpt2_debias', 'gpt2_prompt_tunable']