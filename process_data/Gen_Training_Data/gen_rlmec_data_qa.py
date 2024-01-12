import os
import json
import random

random.seed(42)

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM


ref_model_path = "YOUR_REF_MODEL_PATH"
rw_model_path = "YOUR_GRM_MODEL_PATH"
tokenizer = AutoTokenizer.from_pretrained(ref_model_path)

ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, torch_dtype=torch.bfloat16, device_map='auto')
ref_model.eval()

rw_model = AutoModelForCausalLM.from_pretrained(rw_model_path, torch_dtype=torch.bfloat16, device_map='auto')
ref_model.eval()


def get_prob(input_text, output_text, model=ref_model, is_return_log_prob=True, is_correct=None):
    with torch.no_grad():
        label_ids = tokenizer(output_text + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        input_ids = tokenizer(input_text + output_text + tokenizer.eos_token)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0).to('cuda')

        outputs = model(input_ids)
        
        logits = outputs.logits
        labels = input_ids
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        CEFunc = CrossEntropyLoss(reduction='none')
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)

        log_prob = -CEFunc(shift_logits, shift_labels)
        log_prob = log_prob[-len(label_ids):]
        if (is_return_log_prob == False):
            result = torch.exp(log_prob)
            if (is_correct == True):
                result = torch.clip(result - 0.5, 0, 0.5)
            elif (is_correct == False):
                result = torch.clip(result - 0.5, -0.1, 0)
            else:
                assert(False)
        else:
            result = log_prob

        return result.tolist()



src_data_folder = "../Minimum_Edit/result/qa_grm"
tgt_data_path = "result/rlmec_qa.jsonl"

data = []
files = os.listdir(src_data_folder)
for f in files:
    with open(os.path.join(src_data_folder, f), 'r') as fin:
        tmp_data = fin.readlines()
        data = data + tmp_data
data = [json.loads(d) for d in data]
print(len(data))

pattern = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n### Response: Let's think step by step.\n"
)

REWARD_PROMPT_DICT = {
    "prompt_no_input": 
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
}

REWARD_INSTRUCT = {
    'eval_step': "Given the problem, correct solution and the prediction from language models. The method in prediction might be different with correct solution, but it is also correct. You need to identify which step of the prediction is the first wrong step, and write down the label of the first wrong step.",
    "fix_solu": "Given the problem and the correct solution, you need to correct the mistakes in prediction to get the correct answer. You should make minimal modifications.",
}

REWARD_INPUT = {
    'eval_step': "Problem: {}\nCorrect solution: {}\nPrediction: {}",
    "fix_solu": "Problem: {}\nCorrect solution: {}\nPrediction: {}",
}

REWARD_RESPONSE = {
    'eval_step': "",
    'fix_solu': "Correct prediction:{}",
}

def get_reward_prompt(d):
    input_text = REWARD_PROMPT_DICT['prompt_no_input'].format(
        REWARD_INSTRUCT['fix_solu'],
        REWARD_INPUT['fix_solu'].format(d['question'].strip(), d['solution'], d['prediction']),
        REWARD_RESPONSE['fix_solu'].format(''),
    )
    output_text = d['prediction']
    input_text = input_text
    return input_text, output_text


fout = open(tgt_data_path, 'w')
num_data = 0
p2s = {}
for d in tqdm(data):
    if (d['is_correct'] == False):
        continue
    
    problem = pattern.format(d['question'].strip())
    
    pred = [-1] + tokenizer(d['prediction'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
    fix_pred = [-1] + tokenizer(d['fix_pred'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
    prob = tokenizer(problem, add_special_tokens=False)['input_ids']
    
    len_pred = len(pred)
    len_fix_pred = len(fix_pred)
    
    try:
        assert(len(prob) + len_pred <= 2000)
        assert(len(prob) + len_fix_pred <= 2000)
    except:
        continue
    
    if (d['error_step'] == -1):
        problem = pattern.format(d['question'].strip())
        solution = d['prediction']
        if (problem not in p2s):
            p2s[problem] = {}
        format_solu = solution.replace('\n', '').replace(' ', '').replace('.', '')
        if (format_solu in p2s[problem]):
            continue
            
        ref_prob = get_prob(problem, d['prediction'])
        input_text, output_text = get_reward_prompt(d)
        reward = get_prob(input_text, output_text, model=rw_model, is_return_log_prob=False, is_correct=True)
        
        reg_label = [0.5 for i in range(len_pred)]
        reg_label = reg_label[1:]
        
        assert(len(reward) == len(pred) - 1)
        assert(len(ref_prob) == len(pred) - 1)
        assert(len(reg_label) == len(pred) - 1)
        new_data = {
            'input': problem,
            'output': d['prediction'],
            'regular': d['prediction'],
            'reward': reward,
            'ref_prob': ref_prob,
            'weight_regular': reg_label,
        }
        
        p2s[problem][format_solu] = new_data
        
        continue
        
    num_data = num_data + 1
        
    assert(len_pred <= 2000)
    assert(len_fix_pred <= 2000)
    
    f = [[i for i in range(len_fix_pred)]]
    g = [[(0, max(0, i - 1)) for i in range(len_fix_pred)]]
    for i in range(1, len_pred):
        f.append([i])
        g.append([(max(0, i - 1), 0)])
        for j in range(1, len_fix_pred):
            f[i].append(99999)
            g[i].append((0, 0))
            if (pred[i] == fix_pred[j]):
                f[i][j] = f[i - 1][j - 1]
                g[i][j] = (i - 1, j - 1)
            
            if (f[i - 1][j] + 1 < f[i][j]):
                f[i][j] = f[i - 1][j] + 1
                g[i][j] = (i - 1, j)
            
            if (f[i][j - 1] + 1 < f[i][j]):
                f[i][j] = f[i][j - 1] + 1
                g[i][j] = (i, j - 1)
                
            if (f[i - 1][j - 1] + 1 < f[i][j]):
                f[i][j] = f[i - 1][j - 1] + 1
                g[i][j] = (i - 1, j - 1)
    
    sx, sy = len_pred - 1, len_fix_pred - 1
    reg_label = [0.5 for i in range(len_fix_pred)]
    while (sx != 0 and sy != 0):
        if (pred[sx] != fix_pred[sy]):
            reg_label[sy] = 1.0
        sx, sy = g[sx][sy]
    reg_label = reg_label[1:]
    
    problem = pattern.format(d['question'].strip())
    ref_prob = get_prob(problem, d['prediction'])
    input_text, output_text = get_reward_prompt(d)
    reward = get_prob(input_text, output_text, model=rw_model, is_return_log_prob=False, is_correct=False)
    assert(len(reward) == len(pred) - 1)
    assert(len(ref_prob) == len(pred) - 1)
    assert(len(reg_label) == len(fix_pred) - 1)
    new_data = {
        'input': problem,
        'output': d['prediction'],
        'regular': d['fix_pred'],
        'reward': reward,
        'ref_prob': ref_prob,
        'weight_regular': reg_label,
    }
    fout.write(json.dumps(new_data) + '\n')
    fout.flush()

for prob, solu in p2s.items():
    new_solu = list(solu.items())
    new_solu = random.sample(new_solu, min(2, len(new_solu)))
    for new_data in new_solu:
        fout.write(json.dumps(new_data[1]) + '\n')
        num_data = num_data + 1
    
fout.close()
print(num_data)


