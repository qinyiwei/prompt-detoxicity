#!/bin/sh
python scripts/evaluation/generation/toxicity_analysis.py \
    --generations_file /projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/gpt2/prompted_gens_gpt2.jsonl
python scripts/evaluation/generation/toxicity_analysis.py \
    --generations_file /projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/dapt/prompted_gens_gpt2.jsonl
python scripts/evaluation/generation/toxicity_analysis.py \
    --generations_file /projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/dexperts/large_experts/prompted_gens_dexperts.jsonl
python scripts/evaluation/generation/toxicity_analysis.py \
    --generations_file /projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/gedi/prompted_gens_gedi.jsonl
python scripts/evaluation/generation/toxicity_analysis.py \
    --generations_file /projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/pplm/prompted_gens_pplm.jsonl
#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/generations/toxicity/gpt2/prompted_gens_gpt2.jsonl"
#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/dataset/realtoxicityprompts-data/generations/prompted/prompted_gens_gpt2.jsonl"
#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/self-debiasing/data/generations/toxicity/gpt2/prompted_gens_gpt2.jsonl"
#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/self-debiasing/data/generations/toxicity/self-debias/prompted_generations_gpt2-xl_default.txt"
#orig_input_file = "/projects/tir4/users/yiweiq/toxicity/self-debiasing/data/generations/toxicity/self-debias/prompted_generations_gpt2-xl_debiased.txt"
#orig_input_file = "data/generations/toxicity/gpt2_prompt_tuning/prompted_gens_gpt2_prompt_finetuned_gpt2_owt_loss_type_1_lr_1e-1_checkpoint-1000.jsonl"

