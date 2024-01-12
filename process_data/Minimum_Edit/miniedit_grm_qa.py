import os
import re
import sys
import openai
import json
import time
import random
import string
import argparse

from tqdm import tqdm, trange
from datasets import load_dataset, Dataset, DatasetDict

import sympy
from sympy.parsing.latex import parse_latex


parser = argparse.ArgumentParser()
    
parser.add_argument('--src_data_path', type=str, help='The path to the dataset')
parser.add_argument('--gen_data_folder', type=str, help='The path to the dataset')
parser.add_argument('--result_path', type=str, help='The path to the dataset')
parser.add_argument('--write_mode', type=str, help='The path to the dataset')
parser.add_argument('--start_idx', type=int, help='The path to the dataset')
parser.add_argument('--end_idx', type=int, help='The path to the dataset')
parser.add_argument('--cuda_device', type=str, help='The path to the dataset')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

from vllm import LLM, SamplingParams

stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)

model_path = "YOUR_GRM_PATH"
model = LLM(model_path, dtype='bfloat16')



def call_llm_completion(prompt):
    responses = model.generate([prompt], sampling_params)
    response = responses[0]
    response = response.outputs[0].text
    return response


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


def get_answer_boxed(content):
    pattern = '\\boxed'
    start_pos = content.rfind(pattern)
    if (start_pos == -1): return None
    answer = ''
    num_left = 0
    for i in range(start_pos + 7, len(content)):
        if (content[i] == '}' and num_left == 0):
            break
        if (content[i] == '{'):
            num_left = num_left + 1
        elif (content[i] == '}'):
            num_left = num_left - 1
        answer = answer + content[i]
    return answer

def del_answer_text(content):
    pattern = '\\text'
    start_pos = content.rfind(pattern)
    if (start_pos == -1): return content
    answer = ''
    num_left = 0
    for i in range(start_pos + 6, len(content)):
        if (content[i] == '}' and num_left == 0):
            if (i + 1 < len(content)):
                answer = answer + content[i + 1:]
            break
        if (content[i] == '{'):
            num_left = num_left + 1
        elif (content[i] == '}'):
            num_left = num_left - 1
        answer = answer + content[i]
    if (start_pos > 0):
        answer = content[:start_pos] + answer
    return answer

def test_equal(equ1, equ2):
    try:
        if (equ1.equals(equ2) == True):
            return True
    except:
        pass
    try:
        val_equ1 = sympy.N(equ1)
        val_equ2 = sympy.N(equ2)
        if (abs(float(val_equ1) - float(val_equ2)) <= 0.0001):
            return True
    except:
        pass
    return False

def extract_answer2(Answer_Texts, label):
    if ('\\boxed' in Answer_Texts):
        Answer_Texts = get_answer_boxed(Answer_Texts)
    if ('theansweris' in Answer_Texts.lower()):
        Answer_Texts = Answer_Texts.lower().split('theansweris')[-1].strip()
    
    Answer_Texts = del_answer_text(Answer_Texts)
    label = del_answer_text(label)

    # xxx(round to near ...)
    round_pattern = '\(roundedto.*?\)'
    round_results = re.findall(round_pattern, Answer_Texts)
    for rr in round_results:
        if (rr in Answer_Texts):
            Answer_Texts = Answer_Texts.lower().split(rr)[0].strip()
    
    # approximately xxx
    if ('approximately' in Answer_Texts):
        Answer_Texts = Answer_Texts[Answer_Texts.rfind('approximately') + 1:].strip()
        # Answer_Texts = Answer_Texts.split('approximately')[-1].strip()
    
    # xxx meters
    units = ['meter', 'kilometer', 'kilogram', 'degree', '^\\circ', 'square', 'inches', 'squareunits', 'cm', 'km', 'pound', 'mph', 'hours', 'dollar']
    for unit in units:
        if (unit in Answer_Texts):
            Answer_Texts = Answer_Texts.split(unit)[0].strip()
        if (unit in label):
            label = label.split(unit)[0].strip()
    
    #move un-related symbols
    Answer_Texts = Answer_Texts.replace('\"', '')
    Answer_Texts = Answer_Texts.replace('\'', '')
    label = label.replace('\"', '')
    label = label.replace('\'', '')

    Answer_Texts = Answer_Texts.replace('\\%', '/ 100')
    label = label.replace('\\%', '/ 100')
    if ('$or$' in Answer_Texts):
        first_part = Answer_Texts.split('$or$')[0].strip()
        second_part = Answer_Texts.split('$or$')[-1].strip()
        try:
            sp_first = parse_latex(first_part)
            sp_second = parse_latex(second_part)
            sp_label = parse_latex(label)
            if (test_equal(sp_first, sp_second) == True and test_equal(sp_first, sp_label) == True):
                return True
        except:
            pass

    Answer_Texts = Answer_Texts.replace('$', '')
    label = label.replace('$', '')

    try:
        sp_ans = parse_latex(Answer_Texts)
        sp_label = parse_latex(label)
        if (test_equal(sp_ans, sp_label) == True):
            return True
    except:
        pass

    Answer_Texts = Answer_Texts.replace('\\', '')
    Answer_Texts = Answer_Texts.replace(',!', '')
    Answer_Texts = Answer_Texts.replace(',', '')
    label = label.replace('\\', '')
    label = label.replace(',!', '')
    label = label.replace(',', '')
    
    if len(Answer_Texts) > 0:
        if '.' == Answer_Texts[-1]:
            Answer_Texts = Answer_Texts[:-1]
    Answer_Texts = Answer_Texts.replace(' ', '')

    #make 'dfrac'='frac'
    label = label.replace('dfrac', 'frac')
    Answer_Texts = Answer_Texts.replace('dfrac', 'frac')
    
    if Answer_Texts.rfind('=') > 0:
        Answer_Texts = Answer_Texts[Answer_Texts.rfind('=') + 1:]
    
    #1.00 is not equal to 1 problem
    try:
        if float(int(float(Answer_Texts))) - float(Answer_Texts) == 0:
            Answer_Texts = str(int(float(Answer_Texts)))
    except:
        Answer_Texts = Answer_Texts
    
    try:
        if abs(float(Answer_Texts) - float(label)) <= 0.0001:
            Answer_Texts = label
    except:
        Answer_Texts = Answer_Texts
    
    #make -{a}/{b}={-a}/{b}
    def move_reduce_sign(text):
        index=text.find('-')
        if index>=0:
            return '-'+text[:index]+text[index+1:]
        else:
            return text
    def find_nominator(text):
        index=text.find('{')
        index2=text.find('}')
        return text[index+1:index2]
    def find_denominator(text):
        index=text.rfind('{')
        index2=text.rfind('}')
        return text[index+1:index2]

    if 'frac' in Answer_Texts:
        Answer_Texts=move_reduce_sign(Answer_Texts)
        label=move_reduce_sign(label)

    # a cdot b -> ab
    if label.find('cdot')>=0:
        if Answer_Texts.find('cdot')<0:
            label=label.replace('\\cdot','')
    answer_state = True

    if Answer_Texts != label:
        answer_state = False
    # solving {a*b}/{a*c}!={b}/{c} question by turn the fraction into decimal.
    if label.find('\\dfrac')==0:
        try:
            label_float = float(find_nominator(label)) / float(find_denominator(label))
        except:
            label_float = 'Label can not convert to decimal'
        if Answer_Texts.find('\\dfrac')==0:
            try:
                answer_float = float(find_nominator(Answer_Texts)) / float(find_denominator(Answer_Texts))
            except:
                answer_float = 'Answer can not convert to decimal'
        else:
            try:
                #exec('answer_float=Answer_Texts')
                answer_float=float(Answer_Texts)
            except:
                answer_float='Answer can not convert to decimal'

        if answer_float==label_float:
                answer_state=True
    if Answer_Texts.find('\\dfrac')==0:
        try:
            answer_float = float(find_nominator(Answer_Texts)) / float(find_denominator(Answer_Texts))
        except:
            answer_float = 'Answer can not convert to decimal'
        if label.find('\\dfrac')==0:
            try:
                label_float=float(find_nominator(label))/float(find_denominator(label))
            except:
                label_float='Label can not convert to decimal'
        else:
            try:
                label_float = float(label)
            except:
                label_float = 'Label can not convert to decimal'
        if answer_float==label_float:
            answer_state=True
    return answer_state


def load_gen_dataset(gen_data_folder):
    dataset = []
    files = os.listdir(gen_data_folder)
    for file in files:
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


def build_demo(src_dataset):
    demos = {}
    for data in src_dataset:
        if (data['type'] not in demos):
            demos[data['type']] = []
        demos[data['type']].append(data)
    return demos


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


PROMPT_DICT = {
    "prompt_no_input": 
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
}

INSTRUCT = {
    'eval_step': "Given the problem, correct solution and the prediction from language models. The method in prediction might be different with correct solution, but it is also correct. You need to identify which step of the prediction is the first wrong step, and write down the label of the first wrong step.",
    "fix_solu": "Given the problem and the correct solution, you need to correct the mistakes in prediction to get the correct answer. You should make minimal modifications.",
}

INPUT = {
    'eval_step': "Problem: {}\nCorrect solution: {}\nPrediction: {}",
    "fix_solu": "Problem: {}\nCorrect solution: {}\nPrediction: {}",
}

RESPONSE = {
    'eval_step': "",
    'fix_solu': "Correct prediction:{}",
}



def eval_step(problem, format_solu, format_pred):
    eval_step = PROMPT_DICT['prompt_no_input'].format(
        INSTRUCT['eval_step'],
        INPUT['eval_step'].format(problem.strip(), format_solu, format_pred),
        RESPONSE['eval_step'],
    )
    eval_step = eval_step.strip() + '\n'
    response = call_llm_completion(eval_step)
    label = process_label(response)
    
    return label


def complete_solution(problem, solution, prediction, format_pred, error_step):
    pred_list =[]
    for p in format_pred.split('\n'):
        pred = p.strip()
        pred = p[p.find(']') + 1:].strip()
        pred_list.append(pred)
    solu_prompt = '\n'.join(pred_list[:error_step]).strip()
    
    fix_solu = PROMPT_DICT['prompt_no_input'].format(
        INSTRUCT['fix_solu'],
        INPUT['fix_solu'].format(problem.strip(), solution, prediction),
        RESPONSE['fix_solu'].format(solu_prompt),
    )
    fix_solu = fix_solu.strip() + '\n'
    response = call_llm_completion(fix_solu)
    response = solu_prompt.strip() + '\n' + response
    
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
    step = 0
    is_fix = {}
    fout = open(args.result_path, args.write_mode)
    len_gen_dataset = len(gen_dataset)
    for idx in trange(args.start_idx, args.end_idx):
        if (idx >= len_gen_dataset): break
        data = gen_dataset[idx]
        
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
        
        error_step = eval_step(problem, format_solu, format_pred)
        
        correct_solu = complete_solution(problem, solution, prediction, format_pred, error_step)
        if (correct_solu.endswith('.') == True):
            correct_solu = correct_solu[:-1]
        
        is_new_solution_correct = False
        if ('The answer is' in correct_solu):
            new_ans = correct_solu.split('The answer is')[-1].strip()
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

    print(len(is_fix))
    print('Accuracy on Training Set: ', round(num_correct / total_problem * 100, 2))    
    fout.close()





if __name__ == '__main__':
    main(args)
