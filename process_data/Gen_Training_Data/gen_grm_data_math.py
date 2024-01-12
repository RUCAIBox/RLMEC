import json
import random

src_data = "../Gen_Samples/result/math_tm.jsonl"
tgt_data = "result/grm_math.jsonl"

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
    'fix_solu': "Correct prediction:",
}


def split_solution(solution):
    new_solu = ''
    solu_list = solution.split('\n')
    for i in range(len(solu_list)):
        solu_list[i] = f'[{i}] ' + solu_list[i]
    new_solu = '\n'.join(solu_list)
    return new_solu


with open(src_data, 'r') as fin:
    data = fin.readlines()
data = [json.loads(d) for d in data]
print(len(data))

p2s = {}

num_data = 0
fout = open(tgt_data, 'w')
for d in data:
    if (d['is_correct'] == False or d['error_step'] == -1):
        continue

    format_solu = split_solution(d['solution'])
    format_pred = split_solution(d['prediction'])
    
    eval_step = PROMPT_DICT['prompt_no_input'].format(
        INSTRUCT['eval_step'],
        INPUT['eval_step'].format(d['question'].strip(), format_solu, format_pred),
        RESPONSE['eval_step'],
    )
    eval_step = eval_step.strip() + '\n'
    eval_step_ans = 'The first error step is [{}]'.format(d['error_step'])
    
    new_data = {
        'input': eval_step,
        'output': eval_step_ans,
    }
    fout.write(json.dumps(new_data) + '\n')
    

    fix_solu = PROMPT_DICT['prompt_no_input'].format(
        INSTRUCT['fix_solu'],
        INPUT['fix_solu'].format(d['question'].strip(), d['solution'], d['prediction']),
        RESPONSE['fix_solu'],
    )
    fix_solu_ans = d['fix_pred'].strip()
    
    new_data = {
        'input': fix_solu,
        'output': fix_solu_ans,
    }
    fout.write(json.dumps(new_data) + '\n')
    
    num_data = num_data + 1

print(num_data)

threshold = 1.0 * num_data / len(data)
for d in data:
    if (d['is_correct'] == False or d['error_step'] != -1):
        continue
    flag = random.random()
    if (flag >= threshold):
        continue
        
    fix_solu = PROMPT_DICT['prompt_no_input'].format(
        INSTRUCT['fix_solu'],
        INPUT['fix_solu'].format(d['question'].strip(), d['solution'], d['prediction']),
        RESPONSE['fix_solu'],
    )
    fix_solu_ans = d['prediction'].strip()
    
    new_data = {
        'input': fix_solu,
        'output': fix_solu_ans,
    }
    fout.write(json.dumps(new_data) + '\n')
    
    num_data = num_data + 1
print(num_data)
