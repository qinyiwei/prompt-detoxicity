import torch
import numpy as np
import torch.nn.functional as F

class MyEmbedding(torch.nn.Module):

    # this is for prompt tuning

    def __init__(self, embed, n_prefix, n_class):
        super().__init__()
        self.embed = embed
        self.new_embed = torch.nn.Embedding(n_prefix*n_class, embed.embedding_dim)

        # following Lester et al. 2021 in initializing using the top 5000 random vocabs
        indices = np.random.permutation(range(5000))[:n_prefix*n_class]
        init_weight = self.embed.state_dict()["weight"][indices]
        self.new_embed._load_from_state_dict({"weight": init_weight},
                                             "", None, True, [], [], "")

    def forward(self, input):
        return F.embedding(
            input,
            torch.cat([self.embed.weight, self.new_embed.weight], 0),
            self.embed.padding_idx,
            self.embed.max_norm,
            self.embed.norm_type,
            self.embed.scale_grad_by_freq,
            self.embed.sparse)

def load_prompt_model(model, n_prefix, n_class, save_dir, prompt_only):
    set_extra_embeddings(model, n_prefix, n_class)
    state_dict = torch.load(save_dir+"/pytorch_model.bin")
    if prompt_only:
        model.transformer.wte.embed._load_from_state_dict(
                {"weight": state_dict["embed.weight"]}, "", None, True, [], [], "")
        model.transformer.wte.new_embed._load_from_state_dict(
                {"weight": state_dict["new_embed.weight"]}, "", None, True, [], [], "")
    else:
        model.transformer.wte.embed._load_from_state_dict(
                {"weight": state_dict["transformer.wte.embed.weight"]}, "", None, True, [], [], "")
        model.transformer.wte.new_embed._load_from_state_dict(
                {"weight": state_dict["transformer.wte.new_embed.weight"]}, "", None, True, [], [], "")


def set_extra_embeddings(model, n_prefix, n_class):
    model.transformer.set_input_embeddings(
        MyEmbedding(model.transformer.wte, n_prefix, n_class))


def freeze_LM(model):
    for param in model.parameters():
        param.requires_grad = False