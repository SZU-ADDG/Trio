import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.train_utils import seed_all
import argparse
from dataset import SmileDataset, SmileCollator
from torch.utils.data import DataLoader
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT
import time
import datasets
from rdkit import Chem
from utils.train_utils import get_mol
from fragment_utils import reconstruct
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np


def Test(model, tokenizer, max_seq_len, temperature, top_k, stream, rp, kv_cache, is_simulation, device,
         output_file_path):
    model.eval()
    length_penalty = 1.0
    beam_limit = 8
    morphing_lst = [['*c1ccccc1', '*N1CC2(C[C@H]1C(=O)O)SCCS2'], ['*C1Nc2cc(Cl)c(S(N)(=O)=O)cc2S(=O)(=O)N1', '*C1CC2C=CC1C2'],
                    ['*n1c(NC(C)C)nc2cc(Cl)c(Cl)cc21', '*[C@H]1O[C@@H](CO)[C@H](O)[C@@H]1O'], ['*[C@H]1CCN(C(=O)C=C)C1', '*c1cc(OC)cc(OC)c1'],
                    ['*C1(CC#N)CN(S(=O)(=O)CC)C1', '*c1ncnc2[nH]ccc12'], ['*N1CCCC1', '*c1ccc2c(c1)OCCO2'],
                    ['*C[C@H](N)C(=O)O', '*c1ccc(O)c(I)c1'], ['*[C@@H]1[C@@H]2C(=C[C@H](C)C[C@@H]2OC(=O)[C@@H](C)CC)C=C[C@@H]1C', '*[C@@H]1C[C@@H](O)CC(=O)O1'],
                    ['*n1c(Br)nnc1SCC(=O)O', '*C1CC1'], ['*C#C', '*OCCOC']]

    for idx, morphing in enumerate(morphing_lst):
        complete_answer_list = []
        valid_answer_list = []
        for i in tqdm(range(100)):
            if idx != 2 and idx != 4:
                if i < 50:
                    start_sentence = '[BOS]' + morphing[0]
                    end_sentence = morphing[1] + '[EOS]'
                else:
                    start_sentence = '[BOS]' + morphing[1]
                    end_sentence = morphing[0] + '[EOS]'
            else:
                start_sentence = '[BOS]' + morphing[0]
                end_sentence = morphing[1] + '[EOS]'
            start = tokenizer.encode(start_sentence, add_special_tokens=False)
            start = torch.tensor(start).unsqueeze(0).to(device)
            end = tokenizer.encode(end_sentence, add_special_tokens=False)
            end = torch.tensor(end).unsqueeze(0).to(device)
            with torch.no_grad():
                beams = [{"tokens": start, "score": 0.0, "text": tokenizer.decode(start.squeeze(0))}]
                finished_beams = []
                for step in range(max_seq_len):
                    new_beams = []
                    for beam in beams:
                        inference_res = model(beam["tokens"], tokenizer, kv_cache=False)
                        logits, _, _ = inference_res
                        logits = logits[:, -1, :]

                        for token in set(beam["tokens"].tolist()[0]):
                            if logits[0, token] > 0:
                                logits[0, token] /= rp
                            else:
                                logits[0, token] *= rp

                        logits = logits / temperature if temperature != 0.0 else logits
                        log_probs = F.log_softmax(logits, dim=-1)

                        probs = torch.exp(log_probs)  # 转回真实概率
                        max_prob, max_idx = probs.max(dim=-1)
                        if max_prob > 0.95:  # 比如 0.95
                            # 直接使用最大概率的 token，不采样
                            topk_token_ids = max_idx.unsqueeze(0)
                            topk_log_probs = torch.log(max_prob).unsqueeze(0)
                        else:
                            topk_token_ids = torch.multinomial(probs, num_samples=top_k, replacement=False)
                            topk_log_probs = log_probs.gather(-1, topk_token_ids)  # 不需要再 log

                        # topk_log_probs, topk_token_ids = torch.topk(log_probs, min(top_k, logits.size(-1)), dim=-1)

                        topk_log_probs = topk_log_probs.squeeze(0).tolist()
                        topk_token_ids = topk_token_ids.squeeze(0).tolist()
                        for token, prob in list(zip(topk_token_ids, topk_log_probs)):
                            if token == 27:
                                a = 1
                            candidate_tokens = torch.cat([beam["tokens"], torch.tensor([[token]]).to(device)], dim=1)
                            text = beam["text"] + " " + tokenizer.decode([token])
                            candidate_tokens_text = tokenizer.decode(candidate_tokens[0])
                            # 计算生成概率得分（对数空间）
                            gen_score = beam["score"] + prob

                            # 如果生成了中间部分结尾，计算适配评分
                            if text.endswith('[SEP]'):
                                # full_sequence = torch.cat([candidate_tokens, end], dim=1)
                                # full_text = tokenizer.decode(full_sequence[0])
                                # _, loss, _ = model(full_sequence[:, :-1], tokenizer, targets=full_sequence[:, 1:])
                                # adaptation_score = -loss.item()
                                # finished_gen_score = gen_score / (candidate_tokens.size(1) ** length_penalty)
                                #
                                # finished_beams.append({
                                #     "tokens": candidate_tokens,
                                #     "score": finished_gen_score + adaptation_score,
                                #     "gen_score": finished_gen_score,
                                #     "adaptation_score": adaptation_score,
                                #     "text": text,
                                #     "full_text": full_text
                                # })
                                # 继续扩展，不 return，不 break
                                new_beams.append({
                                    "tokens": candidate_tokens,
                                    "score": gen_score,
                                    "text": text
                                })
                            # 如果生成了 [EOS] —— 评估保留，但不再扩展
                            elif text.endswith('[EOS]'):
                                continue
                            # 正常 token：继续扩展
                            else:
                                # total_score = gen_score  # 未到结尾时仅用生成概率
                                new_beams.append({
                                    "tokens": candidate_tokens,
                                    "score": gen_score,
                                    "text": text
                                })

                            # new_beams.append({"tokens": candidate_tokens, "score": total_score, "text": text})

                    if not new_beams:
                        break
                    for x in new_beams:
                        x["lp_score"] = x["score"] / (x["tokens"].size(1) ** length_penalty)
                    if len(new_beams) > beam_limit:
                        new_beams = sorted(new_beams, key=lambda x: -x["lp_score"])[:beam_limit]
                    for nb in new_beams:
                        if not nb["text"].endswith("[SEP]"):
                            continue
                        candidate_tokens = nb["tokens"]
                        full_sequence = torch.cat([candidate_tokens, end], dim=1)
                        full_text = tokenizer.decode(full_sequence[0])
                        _, loss, _ = model(full_sequence[:, :-1], tokenizer, targets=full_sequence[:, 1:])
                        adaptation_score = -loss.item()
                        finished_gen_score = nb["score"] / (candidate_tokens.size(1) ** length_penalty)

                        finished_beams.append({
                            "tokens": candidate_tokens,
                            "score": finished_gen_score + adaptation_score,
                            "gen_score": finished_gen_score,
                            "adaptation_score": adaptation_score,
                            "text": nb["text"],
                            "full_text": full_text
                        })
                        # scores = torch.tensor([beam["score"] for beam in new_beams], dtype=torch.float32)
                        # probs = torch.softmax(scores, dim=0)
                        # # 采样 beam_limit 个 beam，不放回
                        # selected_indices = torch.multinomial(probs, num_samples=beam_limit, replacement=False).tolist()
                        # new_beams = [new_beams[i] for i in selected_indices]

                    beams = new_beams
            if len(finished_beams) == 0:
                print("No finished beams to sample from!")
                continue

            # Step 1: 提取所有 score
            scores = torch.tensor([beam["score"] for beam in finished_beams], dtype=torch.float32)

            # Step 2: softmax 得到概率分布
            probs = torch.softmax(scores, dim=0)

            # Step 3: 根据概率分布进行加权采样，得到一个 index
            selected_idx = torch.multinomial(probs, num_samples=1).item()

            # Step 4: 取出采样到的 beam
            selected_beam = finished_beams[selected_idx]
            complete_answer = selected_beam["text"] + end_sentence
            complete_answer = complete_answer.replace(" ", "").replace("[BOS]", "").replace("[EOS]", "")
            frag_list = complete_answer.replace(" ", "").split('[SEP]')
            try:
                frag_mol = [Chem.MolFromSmiles(s) for s in frag_list]
                mol = reconstruct(frag_mol)[0]
                if mol:
                    generate_smiles = Chem.MolToSmiles(mol)
                    print("\n", generate_smiles)
                    valid_answer_list.append(generate_smiles)
                    answer = frag_list
                else:
                    answer = frag_list
            except:
                answer = frag_list
            complete_answer_list.append(answer)
            print(f"valid ratio:{len(valid_answer_list)}/{len(complete_answer_list)}={len(valid_answer_list) / len(complete_answer_list)}")
            print(f"uniqueness: {len(list(set(valid_answer_list)))}/{len(valid_answer_list)}={len(list(set(valid_answer_list))) / len(valid_answer_list)}")
        os.makedirs(output_file_path, exist_ok=True)
        with open(os.path.join(output_file_path, f'complete_answer_{idx}.txt'), "w") as w:
            for j in complete_answer_list:
                if not isinstance(j, str):
                    j = str(j)
                w.write(j)
                w.write("\n")
        w.close()
        with open(os.path.join(output_file_path, f'valid_answer_{idx}.txt'), "w") as w:
            for j in valid_answer_list:
                w.write(j)
                w.write("\n")
        w.close()

        a = 1


def main_test(args):
    # 设置随机种子的值
    seed_value = 44
    seed_all(seed_value)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device(f'cuda:{0}')  # 逻辑编号 cuda:0 对应 os.environ["CUDA_VISIBLE_DEVICES"]中的第一个gpu

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
    Test(model, tokenizer, max_seq_len=256, temperature=0.7, top_k=4, stream=False, rp=1.2,
         kv_cache=True, is_simulation=True, device=device, output_file_path="./output/sca_test_44/")
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"运行时间: {elapsed_time:.4f} 秒")


if __name__ == '__main__':
    """
        world_size: 所有的进程数量
        rank: 全局的进程id
    """
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    opt = parser.parse_args()
    world_size = opt.world_size

    main_test(opt)

