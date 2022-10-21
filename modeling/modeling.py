from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers import GPT2LMHeadModel, LogitsProcessorList, LogitsProcessor
from transformers.generation_utils import GenerationMixin, SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from modeling.modeling_add import set_extra_embeddings

class ModelWrapper(ABC):
    """
    This class represents a wrapper for a pretrained language model that provides some high-level functions, including zero-shot
    classification using cloze questions and the generation of texts with self-debiasing.
    """

    def __init__(self, use_cuda: bool = True):
        """
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = None  # type: Optional[PreTrainedTokenizer]
        self._model = None  # type: Optional[PreTrainedModel]
        self.lsm = nn.LogSoftmax(dim=1)

    def query_model(self, input_text: str) -> torch.FloatTensor:
        """For a given input text, returns the probability distribution over possible next tokens."""
        return self.query_model_batch([input_text])[0]

    @abstractmethod
    def query_model_batch_dis(self, input_texts: List[str]) -> torch.FloatTensor:
        """For a batch of input texts, returns the probability distribution over possible next tokens."""
        pass

    @abstractmethod
    def query_model_batch_gen(self, input_texts: List[str]) -> torch.FloatTensor:
        """For a batch of input texts, returns the probability distribution over possible next tokens."""
        pass

    @abstractmethod
    def generate(self, input_text: str, **kwargs) -> str:
        """Generates a continuation for a given input text."""
        pass

    @abstractmethod
    def generate_self_debiasing(self, input_texts: List[str], debiasing_prefixes: List[str], decay_constant: float = 50,
                                epsilon: float = 0.01, debug: bool = False, **kwargs) -> List[str]:
        """
        Generates continuations for the given input texts with self-debiasing.
        :param input_texts: the input texts to generate continuations for
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param kwargs: further arguments are passed on to the original generate function
        :return: the list of generated continuations
        """
        pass

    @abstractmethod
    def compute_loss(self, input_ids: torch.LongTensor, labels: torch.LongTensor) -> torch.Tensor:
        """Computes cross-entropy loss for the given input ids and corresponding labels."""
        pass

    @abstractmethod
    def compute_loss_self_debiasing(self, input_ids: torch.Tensor, trg_len: int, debiasing_prefixes: List[str], decay_constant: float = 50,
                                    epsilon: float = 0.01, debug: bool = False) -> torch.Tensor:
        """
        Computes cross-entropy loss for the given input ids with self-debiasing.
        :param input_ids: the input ids
        :param trg_len: only the last trg_len tokens are considered for computing the loss
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :return: the cross entropy loss
        """
        pass

    def get_token_probability_distribution(self, classfier_type:str,
        input_texts: List, output_texts: List, pretraining_type:str = None):
        """
        For a batch of input texts, returns the probability distribution over possible next tokens considering only the given list of
        output choices.
        :param input_texts: the input texts
        :param output_choices: the allowed output choices (must correspond to single tokens in the model's vocabulary)
        :return: a list of lists, where output[i][j] is a (output, probability) tuple for the ith input and jth output choice.
        """
        result = []
        '''
        if classfier_type == "discriminative":
            output_choice_ids = []
            kwargs = {'add_prefix_space': True} if isinstance(self, GPT2Wrapper) else {}
            for word in output_texts:
                tokens = self._tokenizer.tokenize(word, **kwargs)
                assert len(tokens) == 1, f"Word {word} consists of multiple tokens: {tokens}"
                assert tokens[0] not in self._tokenizer.all_special_tokens, f"Word {word} corresponds to a special token: {tokens[0]}"
                token_id = self._tokenizer.convert_tokens_to_ids(tokens)[0]
                output_choice_ids.append(token_id)

            logits = self.query_model_batch(input_texts)
    
            for idx, _ in enumerate(input_texts):
                output_probabilities = logits[idx][output_choice_ids].softmax(dim=0)
                choices_with_probabilities = list(zip(output_texts, (prob.item() for prob in output_probabilities)))
                result.append(choices_with_probabilities)
        '''
        if classfier_type == "discriminative":
            logits = self.query_model_batch_dis(input_texts, pretraining_type=pretraining_type)

            output_choice_ids = []
            kwargs = {'add_prefix_space': True} if isinstance(self, GPT2Wrapper) else {}
            for word in output_texts:
                tokens = self._tokenizer.tokenize(word, **kwargs)
                assert len(tokens) == 1, f"Word {word} consists of multiple tokens: {tokens}"
                assert tokens[0] not in self._tokenizer.all_special_tokens, f"Word {word} corresponds to a special token: {tokens[0]}"
                token_id = self._tokenizer.convert_tokens_to_ids(tokens)[0]
                output_choice_ids.append(token_id)
            
            for idx, _ in enumerate(input_texts):
                output_probabilities = logits[idx][output_choice_ids].softmax(dim=0)
                choices_with_probabilities = list(zip(output_texts, (prob.item() for prob in output_probabilities)))
                result.append(choices_with_probabilities)

        elif classfier_type == "generative":
            prob_yes = self.get_gen_prob(input_texts, output_texts, 0, pretraining_type=pretraining_type)
            prob_no =  self.get_gen_prob(input_texts, output_texts, 1, pretraining_type=pretraining_type)
            for idx, _ in enumerate(output_texts):
                output_probabilities = [prob_yes[idx], prob_no[idx]]
                choices_with_probabilities = list(zip(['Yes','No'], (prob.item() for prob in output_probabilities)))
                result.append(choices_with_probabilities)
        else:
            raise NotImplementedError

        return result

    
class T5Wrapper(ModelWrapper):
    """A wrapper for the T5 model"""

    def __init__(self, model_name: str = "google/t5-v1_1-xl", use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained T5 model (default: "google/t5-v1_1-xl")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._tokenizer = T5Tokenizer.from_pretrained(model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(model_name)
        if use_cuda:
            self._model.parallelize()
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self._model.config.pad_token_id)
    
    def query_model_batch_dis(self, input_texts: List[str], pretraining_type:str = None):
        if pretraining_type == 'masked_model':
            assert all('<extra_id_0>' in input_text for input_text in input_texts)
        output_texts = ['<extra_id_0>'] * len(input_texts)
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_ids = self._tokenizer.batch_encode_plus(output_texts, return_tensors='pt')['input_ids'].to(self._device)

        if pretraining_type == 'masked_model':    
            return self._model(labels=output_ids, **inputs)['logits'][:, 1, :]
        else:
            return self._model(labels=output_ids, **inputs)['logits'][:, 0, :]
    
    def query_model_batch_gen(self, prompt, output_texts: List[str]):
        input_texts = [prompt] * len(output_texts)
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        outputs = self._tokenizer.batch_encode_plus(output_texts, padding=True, return_tensors='pt')#['input_ids'].to(self._device)
        outputs = {key: val.to(self._device) for key, val in outputs.items()}
        
        output = self._model(labels=outputs['input_ids'], **inputs)
        #print("lm_loss:{}".format(output['loss']))
        return outputs, output['logits']

    def get_gen_prob(self,input_texts,output_texts,idx, pretraining_type=None):
        outputs, logits = self.query_model_batch_gen(input_texts[idx], output_texts)
        tgt_tokens = outputs['input_ids']
        #logits = logits[..., :-1, :].contiguous()
        #tgt_tokens = tgt_tokens[..., 1:].contiguous()
        tgt_len = outputs['attention_mask'].sum(dim=1)

        logits = logits.view(-1, self._model.config.vocab_size)
        loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
        loss = loss.view(tgt_tokens.shape[0], -1)
        prob = -loss.sum(dim=1)
        prob = prob/tgt_len 
        #loss_avg = sum(prob/tgt_len)/len(prob)
        #print("my loss:{}".format(loss_avg))
        return prob


    def generate(self, input_text: str, **kwargs):
        assert '<extra_id_0>' in input_text
        input_ids = self._tokenizer.encode(input_text, return_tensors='pt').to(self._device)
        output_ids = self._model.generate(input_ids, **kwargs)[0]
        return self._tokenizer.decode(output_ids)

    def generate_self_debiasing(self, input_texts: List[str], debiasing_prefixes: List[str], decay_constant: float = 50,
                                epsilon: float = 0.01, debug: bool = False, **kwargs) -> List[str]:
        raise NotImplementedError()

    def compute_loss(self, input_ids: torch.LongTensor, labels: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError()

    def compute_loss_self_debiasing(self, input_ids: torch.Tensor, trg_len: int, debiasing_prefixes: List[str], decay_constant: float = 50,
                                    epsilon: float = 0.01, debug: bool = False) -> torch.Tensor:
        raise NotImplementedError()


class GPT2Wrapper(ModelWrapper):

    def __init__(self, model_name: str = "gpt2-xl", use_cuda: bool = True, tuning_type = None, n_prefix = 20, n_class = 2, is_debias=False):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-xl")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if is_debias:
            self._model = SelfDebiasingGPT2LMHeadModel.from_pretrained(model_name)  # type: SelfDebiasingGPT2LMHeadModel
        else:
            self._model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tuning_type = tuning_type
        self.n_prefix = n_prefix
        self.n_class = n_class

        if tuning_type == "prompt_tuning":
            set_extra_embeddings(self._model, n_prefix, n_class)
            state_dict = torch.load(model_name+"/pytorch_model.bin")
            self._model.transformer.wte.embed._load_from_state_dict(
                    {"weight": state_dict["transformer.wte.embed.weight"]}, "", None, True, [], [], "")
            self._model.transformer.wte.new_embed._load_from_state_dict(
                    {"weight": state_dict["transformer.wte.new_embed.weight"]}, "", None, True, [], [], "")

        if use_cuda:
            self._model.parallelize()
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self._model.config.pad_token_id)
        self.loss_fct2 = torch.nn.CrossEntropyLoss(reduction="none")
        #self.loss_fct = nn.NLLLoss() 
        #use this loss function can get exactly the same result as output['loss']
        #doesn't ignore pad token, while reduction, tgt_len also count the pad token
        #when bs=1, there is no pad token, the above 2 loss functions get the same result

    def input_add_prefix(self, batch_orig, prefix_ids, name):
        input_ids = batch_orig['input_ids']
        bs = input_ids.shape[0]
        new_input_ids = torch.cat([prefix_ids.repeat(bs, 1), input_ids], 1)
        new_attention_mask = torch.cat([torch.ones((bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1)
        token_type_ids = torch.cat([torch.zeros((bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1)
        labels=torch.cat([torch.zeros((bs, self.n_prefix), dtype=torch.long), batch_orig["input_ids"]], 1)

        inputs = {
            "input_ids_{}".format(name): new_input_ids,
            "attention_mask_{}".format(name): new_attention_mask,
            "token_type_ids_{}".format(name): token_type_ids,
            "labels_{}".format(name): labels,
        }

        return inputs

    def get_gen_prob(self,input_texts,output_texts,idx,pretraining_type=None):
        if self.tuning_type is None:
            #tokens = self._tokenizer.tokenize(input_texts[idx])
            model_texts = [input_texts[idx]+text for text in output_texts]
            tokens_prob = self.get_prob([input_texts[idx]], pretraining_type=pretraining_type)
            texts_prob = self.get_prob(model_texts, pretraining_type=pretraining_type)
            return texts_prob - tokens_prob
        elif self.tuning_type == "prompt_tuning":
            batch_orig = self._tokenizer(output_texts, padding = True, return_tensors="pt")

            token_ids= self._tokenizer("".join(input_texts[idx]), return_tensors="pt")["input_ids"]
            assert token_ids.shape[-1] == self.n_prefix
            
            #prepare input
            input_ids = batch_orig['input_ids']
            bs = input_ids.shape[0]
            new_input_ids = torch.cat([token_ids.repeat(bs, 1), input_ids], 1).to(self._device)
            new_attention_mask = torch.cat([torch.ones((bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1).to(self._device)
            token_type_ids = torch.cat([torch.zeros((bs, self.n_prefix), dtype=torch.long), batch_orig["attention_mask"]], 1).to(self._device)
            labels=torch.cat([torch.zeros((bs, self.n_prefix), dtype=torch.long), batch_orig["input_ids"]], 1).to(self._device)

            #caculate loss
            outputs = self._model(input_ids=new_input_ids, attention_mask=new_attention_mask)
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            label_mask = token_type_ids[..., 1:].contiguous()

            losses = self.loss_fct2(logits.view(-1, logits.size(-1)),
                            labels.view(-1)) # [batch_size, length]
            losses = losses.view(logits.size(0), logits.size(1)) * label_mask
            losses = torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)
            return -losses
        else:
            raise NotImplementedError


    def get_prob(self, texts, pretraining_type=None):
        inputs, logits = self.query_model_batch_gen(texts, pretraining_type=pretraining_type)
        tgt_tokens = inputs['input_ids']
        logits = logits[..., :-1, :].contiguous()
        tgt_tokens = tgt_tokens[..., 1:].contiguous()
        tgt_len = inputs['attention_mask'].sum(dim=1)-1

        logits = logits.view(-1, self._model.config.vocab_size)
        loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
        loss = loss.view(tgt_tokens.shape[0], -1)
        prob = -loss.sum(dim=1) 
        prob = prob/tgt_len
        #loss_avg = sum(prob/tgt_len)/len(prob)
        #print("my loss:{}".format(loss_avg))

        return prob

    def query_model_batch_dis(self, input_texts: List[str], pretraining_type:str = None):
        assert pretraining_type=='langauge_model','GPT models only support langauge_model pretrinaing objective'
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_indices = inputs['attention_mask'].sum(dim=1) - 1
        output = self._model(**inputs)['logits']
        return torch.stack([output[example_idx, last_word_idx, :] for example_idx, last_word_idx in enumerate(output_indices)])
    
    
    def query_model_batch_gen(self, input_texts: List[str], pretraining_type:str = None):
        assert pretraining_type=='langauge_model','GPT models only support langauge_model pretrinaing objective'
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output = self._model(**inputs,labels=inputs['input_ids'])
        #print("lm_loss:{}".format(output['loss']))
        return inputs, output['logits']

    def generate(self, input_text: str, **kwargs):
        input_ids = self._tokenizer.encode(input_text, return_tensors='pt').to(self._device)
        output_ids = self._model.generate(input_ids, **kwargs)[0]
        return self._tokenizer.decode(output_ids)

    def generate_self_debiasing(self, input_texts: List[str], debiasing_prefixes: List[str], decay_constant: float = 50,
                                epsilon: float = 0.01, debug: bool = False, min_length: int = None, max_length: int = None,
                                **kwargs) -> List[str]:

        self._model.init_logits_processor(num_debiasing_prefixes=len(debiasing_prefixes), decay_constant=decay_constant, epsilon=epsilon,
                                          debug=debug, tokenizer=self._tokenizer)
        inputs = input_texts.copy()
        for debiasing_prefix in debiasing_prefixes:
            for input_text in input_texts:
                inputs += [debiasing_prefix + input_text]

        inputs = self._tokenizer.batch_encode_plus(inputs, padding=True, return_tensors='pt')
        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = inputs['attention_mask'].shape[-1] - inputs['attention_mask'].sum(dim=-1)
        for batch_idx in range(inputs['input_ids'].shape[0]):
            inputs['input_ids'][batch_idx] = inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(**inputs, min_length=min_length, max_length=max_length, **kwargs)

        batch_size = output_ids.shape[0] // (1 + len(debiasing_prefixes))
        output_ids = output_ids[:batch_size, inputs['input_ids'].shape[1]:]
        return self._tokenizer.batch_decode(output_ids)

    def compute_loss(self, input_ids: torch.LongTensor, labels: torch.LongTensor) -> torch.Tensor:
        outputs = self._model(input_ids, labels=labels)
        lm_logits = outputs[1]

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def compute_loss_self_debiasing(self, input_ids: torch.Tensor, trg_len: int, debiasing_prefixes: List[str], decay_constant: float = 50,
                                    epsilon: float = 0.01, debug: bool = False) -> torch.Tensor:

        self._model.init_logits_processor(num_debiasing_prefixes=len(debiasing_prefixes), decay_constant=decay_constant, epsilon=epsilon,
                                          debug=debug, tokenizer=self._tokenizer)

        input_prefixes = [''] + debiasing_prefixes
        input_prefixes = self._tokenizer.batch_encode_plus(input_prefixes, padding=True, return_tensors='pt')
        input_prefixes['attention_mask'] = torch.flip(input_prefixes['attention_mask'], dims=[1])

        shifts = input_prefixes['attention_mask'].shape[-1] - input_prefixes['attention_mask'].sum(dim=-1)
        for batch_idx in range(input_prefixes['input_ids'].shape[0]):
            input_prefixes['input_ids'][batch_idx] = input_prefixes['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        input_prefixes = {k: v.to(self._device) for k, v in input_prefixes.items()}

        input_ids_repeated = input_ids.repeat(len(debiasing_prefixes) + 1, 1)
        attention_mask = torch.ones_like(input_ids_repeated)

        attention_mask = torch.cat([input_prefixes['attention_mask'], attention_mask], dim=-1)
        input_ids_repeated = torch.cat([input_prefixes['input_ids'], input_ids_repeated], dim=-1)

        target_ids = input_ids_repeated.clone()
        trg_len += shifts[0]
        target_ids[:, :-trg_len] = -100

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self._model(input_ids=input_ids_repeated, attention_mask=attention_mask, position_ids=position_ids, labels=target_ids)
        lm_logits = outputs[1]

        for idx in range(lm_logits.shape[1]):
            lm_logits[:, idx, :] = self._model.logits_processor(input_ids=None, scores=lm_logits[:, idx, :])

        batch_size = lm_logits.shape[0] // (1 + len(debiasing_prefixes))
        lm_logits = lm_logits[:batch_size, shifts[0]:, :]
        target_ids = target_ids[:batch_size, shifts[0]:]

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


class SelfDebiasingLogitsProcessor(LogitsProcessor):
    """This class represents a logits processor that applies self-debiasing."""

    def __init__(self, num_debiasing_prefixes: int, decay_constant: float = 50, epsilon: float = 0.01, debug: bool = False,
                 tokenizer: Optional[PreTrainedTokenizer] = None):
        """
        :param num_debiasing_prefixes: the number of debiasing prefixes used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param tokenizer: a tokenizer used to print debugging output
        """
        assert not debug or tokenizer, "If debug=True, a tokenizer must be passed to SelfDebiasingLogitsProcessor()"
        self.num_debiasing_prefixes = num_debiasing_prefixes
        self.decay_constant = decay_constant
        self.epsilon = epsilon
        self.debug = debug
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0] // (1 + self.num_debiasing_prefixes)
        regular_sentence_indices = range(batch_size)
        for regular_sentence_idx in regular_sentence_indices:
            bias_indices = self._get_bias_indices(regular_sentence_idx, batch_size)
            if bias_indices:
                self._debias_scores(scores, regular_sentence_idx, bias_indices)
        return scores

    def _get_bias_indices(self, regular_sentence_idx: int, batch_size: int) -> List[int]:
        """Returns the indices of all self-debiasing inputs for a regular input"""
        return [regular_sentence_idx + (prefix_idx + 1) * batch_size for prefix_idx in range(self.num_debiasing_prefixes)]

    def _debias_scores(self, scores: torch.FloatTensor, regular_sent_idx: int, bias_indices: List[int]) -> None:
        """Partially debiases the given scores considering a single sentence and the corresponding self-debiasing inputs"""
        logits_biased = [scores[bias_idx] for bias_idx in bias_indices]

        mask = self._generate_decay_mask(scores[regular_sent_idx], logits_biased)
        scores[regular_sent_idx] = torch.log(self._apply_decay_mask(scores[regular_sent_idx], mask))

        for debiasing_sent_idx in bias_indices:
            scores[debiasing_sent_idx] = scores[regular_sent_idx]

    def _apply_decay_mask(self, logits: torch.Tensor, decay_mask: torch.Tensor) -> torch.Tensor:
        """Applies exponential decay to a tensor of logits"""
        probabilities = logits.softmax(dim=-1)
        decay_mask = torch.exp(- decay_mask * self.decay_constant)
        decay_mask = torch.max(decay_mask, torch.tensor([self.epsilon], device=decay_mask.device))
        probabilities = probabilities * decay_mask
        probabilities = probabilities / probabilities.sum(dim=-1)
        return probabilities

    def _generate_decay_mask(self, logits_regular: torch.FloatTensor, logits_biased_list: List[torch.FloatTensor]) -> torch.Tensor:
        """Computes the alpha values (see paper) for each token and stores them in a mask tensor"""
        p_regular = logits_regular.softmax(dim=-1)
        p_biased = None

        for logits_biased in logits_biased_list:
            if p_biased is None:
                p_biased = logits_biased.softmax(dim=-1)
            else:
                p_biased = torch.max(p_biased, logits_biased.softmax(dim=-1))

        if self.debug:
            print(f'== Before Debiasing ==\n'
                  f'Top 5 predictions (regular): {self._get_most_likely_tokens(p_regular, k=5)}\n'
                  f'Top 5 predictions (biased): {self._get_most_likely_tokens(p_biased, k=5)}')

        mask = torch.max(p_biased - p_regular, torch.tensor([0.], device=p_regular.device))

        if self.debug:
            p_regular = self._apply_decay_mask(logits_regular, mask)
            print(f'== After Debiasing ==\n'
                  f'Top 5 predictions (regular): {self._get_most_likely_tokens(p_regular, k=5)}')

        return mask

    def _get_most_likely_tokens(self, probabilities_tensor: torch.Tensor, k: int) -> List[Tuple[str, float]]:
        """Returns the most likely tokens according to a tensor of probabilities"""
        assert len(probabilities_tensor.shape) == 1
        values, indices = torch.topk(probabilities_tensor, k=k, dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(indices)
        return list(zip(tokens, [pv.item() for pv in values]))


class SelfDebiasingGPT2LMHeadModel(GPT2LMHeadModel, GenerationMixin):
    """
    This class represents a regular GPT2LMHeadModel that additionally has the capacity to perform self-debiasing. For self-debiasing, the
    init_logits_processor function must be called. Otherwise, this model just performs regular language modeling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_processor = None  # type: Optional[SelfDebiasingLogitsProcessor]

    def init_logits_processor(self, *args, **kwargs):
        """Initialize the logits processor. For a list of arguments, see the self-debiasing logit processor's init function."""
        self.logits_processor = SelfDebiasingLogitsProcessor(*args, **kwargs)

    def _get_logits_processor(self, *args, **kwargs) -> LogitsProcessorList:
        logits_processor = super()._get_logits_processor(*args, **kwargs)
        if self.logits_processor is not None:
            logits_processor.append(self.logits_processor)
        return logits_processor

    def beam_sample(self, *args, **kwargs):
        raise NotImplementedError("Beam sampling is not implemented for self-debiasing models")

    def sample(self, input_ids: torch.LongTensor, logits_processor: Optional[LogitsProcessorList] = None,
               logits_warper: Optional[LogitsProcessorList] = None, max_length: Optional[int] = None, pad_token_id: Optional[int] = None,
               eos_token_id: Optional[int] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
               output_scores: Optional[bool] = None, return_dict_in_generate: Optional[bool] = None, **model_kwargs) -> Union[
        SampleOutput, torch.LongTensor]:
        """
        This is a verbatim copy of the original implementation by huggingface, with a single modification to ensure that a text and all
        corresponding self-debiasing inputs always chose the same token to generate next. This modification is enclosed by the texts
        "BEGIN MODIFICATIONS" and "END MODIFICATIONS", respectively.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )

        # auto-regressive generation
        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits) #next_token_logits:[bs, #words]
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # =========================
            # BEGIN MODIFICATIONS
            # the following modification to the sample method is necessary to ensure that each debiasing sentence is continued in the same
            # way as the original sentence
            if self.logits_processor is not None:
                batch_size = next_tokens.shape[0] // (1 + self.logits_processor.num_debiasing_prefixes)
                regular_sentence_indices = range(batch_size)
                for regular_sentence_idx in regular_sentence_indices:
                    debiasing_sentence_indices = self.logits_processor._get_bias_indices(regular_sentence_idx, batch_size)
                    for debiasing_sentence_idx in debiasing_sentence_indices:
                        next_tokens[debiasing_sentence_idx] = next_tokens[regular_sentence_idx]
            # END MODIFICATIONS
            # =========================

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids