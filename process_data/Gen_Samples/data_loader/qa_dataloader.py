import json
import random

from datasets import Dataset


class qa_dataloader:
    def __init__(self, dataset_path, num_exemplars=0):
        self.dataset_path = dataset_path
        self.num_exemplars = num_exemplars
        self.pattern = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{}\n\n### Response: Let's think step by step.\n"
        )
        
        self.train_dataset = self.load_jsonl_data(self.dataset_path)
        
    def load_jsonl_data(self, dataset_path):
        with open(dataset_path, 'r') as fin:
            raw_dataset = fin.readlines()
            raw_dataset = [json.loads(d) for d in raw_dataset]
        dataset = {}
        for data in raw_dataset:
            for k in data.keys():
                if (k not in dataset):
                    dataset[k] = []
                dataset[k].append(data[k])
        return Dataset.from_dict(dataset)
    
    def load_train_data(self):
        return self.train_dataset
    
    def load_demo(self):
        return ''
    
    def process_dataset(self, dataset):
        processed_dataset = {'processed_data': []}
        for data in dataset:
            processed_dataset['processed_data'].append(
                self.pattern.format(data['problem'])
            )
        return Dataset.from_dict(processed_dataset)
    
    def clean_exemplar(self, content):
        query = content.split('### Response:')[self.num_exemplars].strip().split('### Instruction:')[-1].strip()
        return query

