import torch
import os
import argparse
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
from mcts import MCTSConfig, MolecularProblemState, MCTS
from rdkit import rdBase
from utils.docking.docking_utils import DockingVina
import pandas as pd
from tqdm import tqdm

rdBase.DisableLog('rdApp.warning')


def Test(model, tokenizer, device, output_file_path, sample_num):
    os.makedirs(output_file_path, exist_ok=True)
    model.eval()
    predictor = DockingVina('parp1')
    results = []
    x = torch.tensor([1], dtype=torch.int64).unsqueeze(0)  # [BOS]
    x = x.to(device)
    sample_num = int(sample_num)
    for i in range(sample_num):
        print('sample:', i+1)
        initial_state = MolecularProblemState(model, tokenizer, predictor, x)
        mcts_config = MCTSConfig()
        mcts = MCTS(initial_state, mcts_config)
        with torch.no_grad():
            rv, rq, rs, smi, cur_sentence = mcts.run()
            results.append([rv, rq, rs, smi, cur_sentence])

    df = pd.DataFrame(results, columns=['rv', 'rq', 'rs', 'smi', 'cur_sentence'])
    df.to_csv(os.path.join(output_file_path, f'mcts.csv'), index=False)


def main_test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    device = torch.device(f'cuda:{0}')

    tokenizer = SmilesTokenizer('./vocabs/vocab.txt')
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    mconf = GPTConfig(vocab_size=tokenizer.vocab_size, n_layer=12, n_head=12, n_embd=768)
    model = GPT(mconf).to(device)
    checkpoint = torch.load(f'./weights/fragpt.pt', weights_only=True)

    model.load_state_dict(checkpoint)
    start_time = time.time()
    Test(model, tokenizer, device, output_file_path=args.output_file_path, sample_num=args.sample_num)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"running time: {elapsed_time:.4f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file_path', default='./output/mcts', help='output_csv_path')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--sample_num', default='1', help='number of sample')

    opt = parser.parse_args()

    main_test(opt)

