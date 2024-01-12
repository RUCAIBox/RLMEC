#

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import random
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, apply_rotary_pos_emb

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
        self.sources, self.targets = [], []
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

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.sources[i], labels=self.targets[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    data_args: DataArguments
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if self.data_args.dailog_augmentation == False:
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

            return inputs
        else:
            input_ids_list, labels_list = [], []
            max_length = self.tokenizer.model_max_length
            for instance in instances:
                raw_text = instance['labels'].rstrip(self.tokenizer.eos_token)
                text = []
                for i, txt in enumerate(raw_text.split('\n[|AI|]:')):
                    if i == 0:
                        text.append(txt + '\n[|AI|]:')
                    else:
                        split_txt = txt.split('\n[|Human|]:')
                        ai_txt = split_txt[0]
                        text.append(ai_txt + self.tokenizer.eos_token)
                        if len(split_txt) == 2:
                            human_txt = split_txt[1]
                            text.append('\n[|Human|]:' + human_txt + '\n[|AI|]:')
                inputs = self.tokenizer(text=text, max_length=max_length, truncation=True)
                input_ids, labels = [], []
                for i, iids in enumerate(inputs['input_ids']):
                    if i != 0:
                        iids = iids[1:]
                    input_ids.extend(iids)
                    if i % 2 == 0:
                        labels.extend([IGNORE_INDEX] * len(iids))
                    else:
                        labels.extend(iids)
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                labels = torch.tensor(labels, dtype=torch.long)
                input_ids_list.append(input_ids[:max_length])
                labels_list.append(labels[:max_length])
                assert len(input_ids_list[-1]) == len(labels_list[-1])
                if len(input_ids_list[-1]) > 2048:
                    print(raw_text)
                    print(input_ids)
                    print(labels)
                    exit(0)
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
            inputs = {
                'input_ids': input_ids,
                'labels': labels,
            }
            assert input_ids.shape == labels.shape
            if input_ids.size(1) > 2048:
                print(instances)
                print(inputs)
                exit(0)
            return inputs


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, prompt_type=data_args.prompt_type, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(data_args=data_args, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
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
