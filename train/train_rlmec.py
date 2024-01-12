import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import random
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, apply_rotary_pos_emb
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"
PROMPT_DICT = {
    "simple_inference": (
        "{input}"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attention: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    prompt_type: Optional[str] = field(default="instruction")
    dailog_augmentation: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, prompt_type: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        prompt_simple_inference = PROMPT_DICT["simple_inference"]
        self.sources, self.targets, self.reward, self.ref_prob, self.regular_sent, self.weight_reg = [], [], [], [], [], []
        for path in data_path.split(','):
            with open(path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    try:
                        c = json.loads(line)
                    except:
                        print(path)
                        print(line)
                        raise ValueError

                    input_text = c['input']
                    source = prompt_simple_inference.format_map(dict(input=input_text))
                    self.sources.append(source.strip())

                    output_text = c['output']
                    self.targets.append(source + output_text + tokenizer.eos_token)

                    self.reward.append(c['reward'])
                    self.ref_prob.append(c['ref_prob'])
                    self.regular_sent.append(source + c['regular'] + tokenizer.eos_token)
                    self.weight_reg.append(c['weight_regular'])

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.sources[i], 
            labels=self.targets[i],
            regular_input_ids=self.regular_sent[i],
            reward=self.reward[i],
            ref_prob=self.ref_prob[i],
            weight_reg=self.weight_reg[i],
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    data_args: DataArguments
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            text=[instance['labels'] for instance in instances],
            text_target=[instance['input_ids'] for instance in instances],
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        labels = copy.deepcopy(inputs['input_ids'])
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        labels[torch.where(inputs['labels'] != self.tokenizer.pad_token_id)] = IGNORE_INDEX
        inputs['labels'] = labels
        
        regular = self.tokenizer(
            text=[instance['regular_input_ids'] for instance in instances],
            text_target=[instance['input_ids'] for instance in instances],
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        regular_labels = copy.deepcopy(regular['input_ids'])
        regular_labels[regular_labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        regular_labels[torch.where(regular['labels'] != self.tokenizer.pad_token_id)] = IGNORE_INDEX
        inputs['regular_input_ids'] = regular['input_ids']
        inputs['regular_labels'] = regular_labels
        inputs['regular_attn_mask'] = regular['attention_mask']
        
        cur_len = inputs['input_ids'].size(1)
        
        reward = [[0] * (cur_len - len(instance['reward'])) + instance['reward'] for instance in instances]
        reward = torch.tensor(reward, dtype=torch.bfloat16)
        inputs['reward'] = reward
        
        ref_prob = [[0] * (cur_len - len(instance['ref_prob'])) + instance['ref_prob'] for instance in instances]
        ref_prob = torch.tensor(ref_prob, dtype=torch.bfloat16)
        inputs['ref_prob'] = ref_prob
        
        
        reg_cur_len = inputs['regular_input_ids'].size(-1)
        
        weight_reg = [[0] * (reg_cur_len - len(instance['weight_reg'])) + instance['weight_reg'] for instance in instances]
        weight_reg = torch.tensor(weight_reg, dtype=torch.bfloat16)
        inputs['weight_reg'] = weight_reg

        return inputs


class LlamaForRLMEC(LlamaForCausalLM):
    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        regular_input_ids,
        regular_labels,
        regular_attn_mask,
        reward,
        ref_prob,
        weight_reg,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        reward = reward[..., 1:].contiguous()
        reward = reward.view(-1).contiguous()
        
        ref_prob = ref_prob[..., 1:].contiguous()
        ref_prob = ref_prob.view(-1).contiguous()
        log_ref_prob = ref_prob
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        CEFunc = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        log_policy_prob = -CEFunc(shift_logits, shift_labels)
        
        epsilon=0.3
        
        num_label = torch.sum(torch.ne(reward, 0.0))
        with torch.no_grad():
            importance_sampling = torch.exp(log_policy_prob - log_ref_prob)
            importance_sampling_clip = torch.clip(importance_sampling, min = 1 - epsilon, max = 1 + epsilon)
            importance_sampling = torch.min(importance_sampling * reward, importance_sampling_clip * reward)
        
        
        loss_ppo = -importance_sampling * log_policy_prob
    
        loss_ppo = loss_ppo.sum() / num_label
        
            
        regular_outputs = self.model(
            input_ids=regular_input_ids,
            attention_mask=regular_attn_mask,
        )
        hidden_states = regular_outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = regular_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss_regular = loss_fct(shift_logits, shift_labels)
        
        weight_reg = weight_reg[..., 1:].contiguous()
        weight_reg = weight_reg.view(-1).contiguous()
        num_label = torch.sum(torch.ne(weight_reg, 0.0))
        loss_regular = loss_regular * weight_reg
        loss_regular = loss_regular.sum() / num_label
        
        
        lamb = 1.0
        loss = loss_ppo + lamb * loss_regular

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, prompt_type=data_args.prompt_type, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(data_args=data_args, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = LlamaForRLMEC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
    )
    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    assert num_new_tokens == 0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
