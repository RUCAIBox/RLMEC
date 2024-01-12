import argparse
import os
import json
import random

from tqdm import tqdm
from vllm import LLM, SamplingParams


random.seed(42)

pattern = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n### Correct Response: Let's think step by step.\n"
)


def load_jsonl_data(data_path):
    with open(data_path, 'r') as fin:
        dataset = fin.readlines()
        dataset = [json.loads(d) for d in dataset]
    return dataset


def load_demo(dataset, n_shot):
    if (n_shot == 0):
        return ""
    exemplars = random.sample(dataset, n_shot)
    prompt = ""
    for e in exemplars:
        prompt = prompt + pattern.format(e['problem']) + e['solution']
        prompt = prompt.strip() + '\n\n'
    prompt = prompt.strip() + '\n\n'
    return prompt


def call_llm_completion(model, sampling_params, prompt):
    responses = model.generate([prompt], sampling_params)
    response = responses[0]
    response = response.outputs[0].text
    return response


def main(args):
    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", 'Below']
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512, stop=stop_tokens)
    model = LLM(model=args.model)
    
    train_dataset = load_jsonl_data(args.train_file)
    test_dataset = load_jsonl_data(args.dev_file)
    
    demo = load_demo(train_dataset, args.n_shot)
    
    num_correct = 0
    total_problem = 0
    fout = open(args.result_file, 'w')
    for data in tqdm(test_dataset):
        prompt = demo + pattern.format(data['problem'])
        
        response = call_llm_completion(model, sampling_params, prompt)
        if ('The answer is' in response):
            pred_ans = response.split('The answer is')[-1].strip().lower()
        else:
            pred_ans = ''
        if (pred_ans == data['answer']):
            num_correct = num_correct + 1
        total_problem = total_problem + 1
        
        new_data = {
            'prompt': prompt,
            'problem': data['problem'],
            'real_ans': data['answer'],
            'prediction': response,
            'pred_ans': pred_ans,
        }
        fout.write(json.dumps(new_data) + '\n')
    fout.close()
    
    print('Accuracy: {} ( {} / {} )'.format(round(num_correct / total_problem * 100, 2), num_correct, total_problem))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str)
    parser.add_argument("--train_file", type=str, default='data/train.jsonl')
    parser.add_argument("--dev_file", type=str, default='data/test.jsonl')
    parser.add_argument("--result_file", type=str, default='')
    parser.add_argument("--n_shot", type=int, default=0)

    args = parser.parse_args()
    
    main(args)
