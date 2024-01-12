import os
import json
import random
import argparse

from tqdm import trange

from data_loader.math_dataloader import math_dataloader
from data_loader.qa_dataloader import qa_dataloader


DATALOADER = {
    'math': math_dataloader,
    'qa': qa_dataloader,
}


def main(args):
    import torch
    from vllm import LLM, SamplingParams

    random.seed(args.seed)

    dataloader = DATALOADER[args.data_name](dataset_path=args.data_path)
    demo = dataloader.load_demo()
    dataset = dataloader.load_train_data()
    dataset = dataloader.process_dataset(dataset)
    num_data = len(dataset)
    print(demo)
    print(dataset)
    print('Num data:', num_data)

    num_gpu = len(args.cuda_device.split(','))
    if (torch.cuda.device_count() != num_gpu):
        raise Warning('CUDA may be not correctly set.')
    
    model = LLM(args.model_path, dtype='bfloat16')
    sampling_params = SamplingParams(top_p=0.95, temperature=0.7, n=5, max_tokens=512)
    print(model)
    print(sampling_params)

    folder_path = args.target_path.rsplit('/')[:-1]
    folder_path = '/'.join(folder_path)
    print('Result Folder:', folder_path)
    if (os.path.exists(folder_path) == False):
        os.mkdir(folder_path)
    fout = open(args.target_path, args.write_mode)

    def make_query(input_text):
        nonlocal fout
        responses = model.generate(input_text, sampling_params)
        for response in responses:
            query = response.prompt
            query = dataloader.clean_exemplar(query)
            response = response.outputs
            response = [r.text for r in response]
            for r in response:
                for w in ['Problem:', '</s>', 'Human:', 'Assistant:']:
                    if (w in r): r = r.split(w)[0].strip()
                result = {
                    'question': query,
                    'prediction': r,
                }
                fout.write(json.dumps(result) + '\n')
                fout.flush()

    input_data = []
    for idx in trange(args.start_idx, args.end_idx):
        if (idx >= num_data): break
        data = dataset[idx]
        input_text = demo + data['processed_data']
        input_data.append(input_text)
        if (len(input_data) == args.batch_size):
            make_query(input_data)
            input_data = []
    if (len(input_data) != 0):
        make_query(input_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--end_idx", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--target_path", type=str)
    parser.add_argument("--cuda_device", type=str)
    parser.add_argument("--write_mode", type=str)

    args = parser.parse_args()

    print('CUDA device: {}'.format(args.cuda_device))
    print('Data path: {}'.format(args.data_path))
    print('Target path: {}'.format(args.target_path))
    print('Random seed: {}'.format(args.seed))
    print('')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    main(args)
