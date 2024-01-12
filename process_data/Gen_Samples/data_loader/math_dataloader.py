import json

from datasets import Dataset


class math_dataloader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.pattern = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{}\n\n### Response:"
        with open(self.dataset_path, 'r') as fin:
            raw_dataset = fin.readlines()
            raw_dataset = [json.loads(d) for d in raw_dataset]
        self.dataset = {}
        for data in raw_dataset:
            for k in data.keys():
                if (k not in self.dataset):
                    self.dataset[k] = []
                self.dataset[k].append(data[k])
        self.dataset = Dataset.from_dict(self.dataset)
        
    def load_demo(self):
        return ''
    
    def load_train_data(self):
        return self.dataset
    
    def process_dataset(self, dataset):
        processed_dataset = {'processed_data': []}
        for data in dataset:
            processed_dataset['processed_data'].append(
                self.pattern.format(data['query'])
            )
        return Dataset.from_dict(processed_dataset)
    
    def clean_exemplar(self, content):
        query = content.split('### Response:')[0].strip().split('### Instruction:')[-1].strip()
        return query
