import os
import re
import sys
import openai
import json
import time
import random
import string
import argparse
import anthropic

from tqdm import tqdm, trange
from datasets import load_dataset, Dataset, DatasetDict

import sympy
from sympy.parsing.latex import parse_latex


client = anthropic.Client(
    "YOUR_CLAUDE_API_KEY"
)


def call_claude_completion(
    prompt,
    ai_prompt="",
    exemplar="",
    model="claude-2",
    stop=None,
    max_tokens=350,
):
    claude_prompt = exemplar + anthropic.HUMAN_PROMPT + prompt.strip() + anthropic.AI_PROMPT + ai_prompt.strip()
    while True:
        try:
            response = client.completion(
                prompt=claude_prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT],
                model=model,
                max_tokens_to_sample=max_tokens,
                temperature=0,
            )
            break
        except Exception as e:
            time.sleep(1)
    return response["completion"].strip()


def clean(content):
    pattern = '<<.+>>'
    result = re.findall(pattern, content)
    for t in result:
        content = content.replace(t, '')
    answer = content.split('####')[-1].strip()
    content = content.split('####')[0]
    content = content + ' The answer is {}'.format(answer)
    return content


def process_label(label):
    if ('correct' in label.lower()):
        return -1
    for ch in label:
        if (ch.isdigit() == True):
            return int(ch)
    return -1









def load_gen_dataset(gen_data_folder):
    dataset = []
    files = os.listdir(gen_data_folder)
    for file in files:
        if (file.endswith('jsonl') == False):
            continue
        with open(os.path.join(gen_data_folder, file), 'r') as fin:
            tmp_dataset = fin.readlines()
            dataset = dataset + tmp_dataset
    dataset = [json.loads(d) for d in dataset]
    hf_dataset = {}
    for data in dataset:
        for k in data.keys():
            if (k not in hf_dataset):
                hf_dataset[k] = []
            hf_dataset[k].append(data[k])
    return Dataset.from_dict(hf_dataset)


def load_src_dataset(src_data_path):
    with open(src_data_path, 'r') as fin:
        dataset = fin.readlines()
    dataset = [json.loads(d) for d in dataset]
    hf_dataset = {}
    for data in dataset:
        for k in data.keys():
            if (k not in hf_dataset):
                hf_dataset[k] = []
            hf_dataset[k].append(data[k])
    return Dataset.from_dict(hf_dataset)


def build_prob2ans(src_dataset):
    prob2ans = {}
    for data in src_dataset:
        prob2ans[data['problem'].strip()] = {
            'solution': data['solution'],
            'answer': data['answer'],
        }
    return prob2ans



def check_valid(prob2ans, gen_dataset):
    print('Checking prob2ans ...')
    for data in gen_dataset:
        try:
            assert(data['question'].strip() in prob2ans)
        except AssertionError:
            print(data['question'])
            exit(0)
    print('Checking pass!')


def split_solution(solution):
    new_solu = ''
    solu_list = solution.split('\n')
    for i in range(len(solu_list)):
        solu_list[i] = f'[{i}] ' + solu_list[i]
    new_solu = '\n'.join(solu_list)
    return new_solu


pattern_eval_step = 'Given the problem, correct solution and the prediction from language models. The method in prediction might be different with correct solution, but it is also correct. You need to identify which step of the prediction is the first wrong step, and write down the label of the first wrong step.\n\nProblem: {}\n\nCorrect solution: {}\n\nPrediction: {}\n\nWhich step of prediction is error? Only write down the label of the first wrong step. If the prediction is correct, you need to write down correct. You should not write down any other words.'


def eval_step(problem, format_solu, format_pred):
    prompt = pattern_eval_step.format(problem, format_solu, format_pred)
    
    response = call_claude_completion(prompt)
    label = process_label(response)
    print('EVAL_STEP REAPONSE: ', response, '\n')
    
    return label













exemplar_complete = anthropic.HUMAN_PROMPT + "Given the problem and the correct solution, you need to correct the mistakes in prediction to get the correct answer. You should make minimal modifications.\nProblem: Where would you borrow coffee if you do not have any?\nOptions:\nmeeting\nconvenience store\nsupermarket\nfast food restaurant\nfriend's house\nCorrect solution: If you don't have coffee powder / beans and don't want to buy it, you would borrow from friend's house\nFriend will not sell you but will give you out of love.\nThe answer is friend's house\nPrediction: If you need to borrow coffee, it is more likely that you do not have coffee at home\nCoffee can be found in a supermarket\nThe answer is supermarket " + anthropic.AI_PROMPT + "Correct prediction: If you need to borrow coffee, it is more likely that you do not have coffee at home\nCoffee can be found at a friend's house and you can borrow the coffee from your friend\nThe answer is friend's house"

pattern_exemplar = "You need to solve the problem step by step and also refer to the questions you have done in the past.\nProblem: {}"
pattern_complete = "Given the problem and the correct solution, you need to correct the mistakes in prediction to get the correct answer. You should make minimal modifications.\nProblem: {}\nCorrect solution: {}\nPrediction: {}"
pattern_complete_ai = "Correct prediction: {}"



def complete_solution(problem, solution, prediction, format_pred, error_step):
    pred_list =[]
    for p in format_pred.split('\n'):
        pred = p.strip()
        pred = p[p.find(']') + 1:].strip()
        pred_list.append(pred)
    if (error_step >= len(pred_list)):
        return ''
    solu_prompt = '\n'.join(pred_list[:error_step]).strip()

    prompt = pattern_complete.format(problem, solution, prediction)
    ai_prompt = pattern_complete_ai.format(solu_prompt)
    response = call_claude_completion(prompt, exemplar=exemplar_complete, ai_prompt=ai_prompt)
    response = solu_prompt.strip() + '\n' + response
    print('COMPLETE_SOLUTION RESPONSE: ', response)
    
    return response


def main(args):
    gen_dataset = load_gen_dataset(args.gen_data_folder)
    src_dataset = load_src_dataset(args.src_data_path)
    prob2ans = build_prob2ans(src_dataset)
    check_valid(prob2ans, gen_dataset)
    print(gen_dataset)
    print(src_dataset)
    
    if (args.write_mode == 'a'):
        with open(args.result_path, 'r') as fin:
            tmp_data = fin.readlines()
            num_completed = len(tmp_data)
            if (tmp_data[-1] == '\n'):
                num_completed = num_completed - 1
    else:
        num_completed = 0
    
    num_correct = 0
    total_problem = 0
    fout = open(args.result_path, args.write_mode)
    is_fix = {}
    for data in tqdm(gen_dataset):
        problem = data['question'].strip()
        solution = prob2ans[problem]['solution'].strip()
        real_ans = prob2ans[problem]['answer'].strip()
        if ('Below is an instruction' in data['prediction']):
            data['prediction'] = data['prediction'].split('Below is an instruction')[0].strip()
        prediction = data['prediction']
        
        if (prediction in is_fix):
            continue
        is_fix[prediction] = 1
        
        if (num_completed > 0):
            num_completed = num_completed - 1
            continue
        
        if ('The answer is' in prediction):
            pred_ans = prediction.split('The answer is')[-1].strip()
        else:
            pred_ans = ' '
            
        is_ans_correct = False
        if (pred_ans.lower() == real_ans.lower()):
            new_data = {
                'question': problem,
                'solution': solution,
                'prediction': prediction,
                'error_step': -1,
                'error_token': '',
                'correct_step': '',
                'fix_pred': prediction,
                'is_correct': True,
            }
            fout.write(json.dumps(new_data) + '\n')
            fout.flush()
            continue
        format_solu = split_solution(solution.strip())
        format_pred = split_solution(prediction.strip())
        print(problem, '\n')
        print(format_solu, '\n')
        print(format_pred, '\n')
        
        error_step = eval_step(problem, format_solu, format_pred)
        
        correct_solu = complete_solution(problem, solution, prediction, format_pred, error_step)
        if (correct_solu.endswith('.') == True):
            correct_solu = correct_solu[:-1]
        
        is_new_solution_correct = False
        if ('The answer is' in correct_solu):
            new_ans = correct_solu.split('The answer is')[-1].strip()
            print(new_ans, real_ans, new_ans.lower() == real_ans.lower())
            if (new_ans.lower() == real_ans.lower()):
                is_new_solution_correct = True
        

        if (is_new_solution_correct == True):
            num_correct = num_correct + 1
        total_problem = total_problem + 1
        
        new_data = {
            'question': problem,
            'solution': solution,
            'prediction': prediction,
            'error_step': error_step,
            'error_token': '',
            'correct_step': '',
            'fix_pred': correct_solu,
            'is_correct': is_new_solution_correct,
        }
        fout.write(json.dumps(new_data) + '\n')
        fout.flush()
        print('\n--------------------------\n')
        
    print('Accuracy on Training Set: ', round(num_correct / total_problem * 100, 2))    
    fout.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--src_data_path', type=str, help='The path to the dataset')
    parser.add_argument('--gen_data_folder', type=str, help='The path to the dataset')
    parser.add_argument('--result_path', type=str, help='The path to the dataset')
    parser.add_argument('--write_mode', type=str, help='The path to the dataset')

    args = parser.parse_args()

    main(args)
