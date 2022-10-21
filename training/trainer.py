from transformers import  Trainer
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers.optimization import Adafactor, AdamW, get_scheduler
import random
from typing import Optional
import os
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.file_utils import WEIGHTS_NAME

logger = logging.get_logger(__name__)

class ContrastiveTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)
        #self.loss = nn.MarginRankingLoss(margin=self.args.margin)
        self.lsm = torch.nn.CrossEntropyLoss(reduction="none")

    def cal_loss(self, model, input_ids, attention_mask, token_type_ids,
              labels):

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()



        losses = self.lsm(logits.view(-1, logits.size(-1)),
                        labels.view(-1)) # [batch_size, length]
        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        F_maximize = -self.cal_loss(model, inputs['input_ids_maxi'], inputs['attention_mask_maxi'], inputs['token_type_ids_maxi'], inputs['labels_maxi'])
        F_minimize = -self.cal_loss(model, inputs['input_ids_mini'], inputs['attention_mask_mini'], inputs['token_type_ids_mini'], inputs['labels_mini'])
        
        if self.args.loss_type == 1:
            loss = F.relu(F_minimize - F_maximize + self.args.margin)
        elif self.args.loss_type == 2:
            loss = - self.args.loss_alpha * F_maximize + self.args.loss_beta * F.relu(F_minimize - F_maximize + self.args.margin)
        elif self.args.loss_type == 3:
            prob = random.uniform(0, 1)
            if prob > 0.5:
                loss = - F_maximize 
            else:
                loss = F.relu(F_minimize - F_maximize + self.args.margin)
        elif self.args.loss_type == 4: 
            print(self.state.global_step)
            if int(self.state.global_step/100)%2==0:
                loss = - F_maximize 
            else:
                loss = F.relu(F_minimize - F_maximize + self.args.margin)
        else:
            raise NotImplementedError
            
        loss = loss.mean()
        #print("loss:{}".format(loss))
        #print("F_maximize:{}".format(F_maximize.mean()))
        #print("F_minimize:{}".format(F_minimize.mean()))
        
        return loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]

            p1 = [n for n, p in self.model.named_parameters() if 'new_embed' in n and not any(nd in n for nd in no_decay)]
            p2 = [n for n, p in self.model.named_parameters() if 'new_embed' in n and any(nd in n for nd in no_decay)]
            p3 = [n for n, p in self.model.named_parameters() if 'new_embed' not in n and not any(nd in n for nd in no_decay)]
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if 'new_embed' in n],
                    "weight_decay": self.args.weight_decay,
                    "lr":self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if 'new_embed' not in n and not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                    "lr":self.args.learning_rate_LM,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if 'new_embed' not in n and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr":self.args.learning_rate_LM,
                },
            ]
            print(p1)
            print("lr is:{}".format(optimizer_grouped_parameters[0]["lr"]))
            print(p2)
            print("lr is:{}".format(optimizer_grouped_parameters[1]["lr"]))
            print(p3)
            print("lr is:{}".format(optimizer_grouped_parameters[2]["lr"]))

            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
    
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if not self.args.freeze_LM:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.model.transformer.wte.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if not self.args.freeze_LM:
                self.model.save_pretrained(output_dir)
            else:
                state_dict = self.model.transformer.wte.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))