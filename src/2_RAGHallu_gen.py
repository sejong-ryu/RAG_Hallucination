import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import joblib
import logging
from datetime import datetime
from argparse import ArgumentParser
import utils
from utils import normalize_answer, find_subsequence
import json


def set_logger(args):
    current_time = datetime.now()
    log_folder = f"../logs/{args.model}/{args.data}"
    if not os.path.exists(log_folder):
        utils.make_dir(log_folder)
    log_filename = os.path.join(log_folder, f"{args.model}-{args.data}-{current_time.strftime('%Y_%m_%d-%H_%M_%S')}.txt")
    utils.make_dir(log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
        # handlers = [logging.FileHandler(log_filename)]
    )

    logger = logging.getLogger()
    # logger.setLevel(logging.CRITICAL)
    logger.setLevel(logging.INFO)

    return logger



def set_seed(random_seed=2025):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



class AttentionExtractor:
    '''Extract the attention maps of the specific attention heads.'''
    def __init__(self, model, topk_head_index):
        self.model = model
        self.device = self.model.device
        self.topk_head_index = topk_head_index  # [[layer_idx, head_idx], ...]
        self.layer_idx, self.head_idx = None, None
        self.attentions = {(layer, head): None for layer,head in topk_head_index}

    def register_hooks(self):
        for layer_idx, head_idx in self.topk_head_index:
            layer = self.model.model.layers[layer_idx]
            if hasattr(layer, 'self_attn'):
                def hook_wrapper(layer_idx=layer_idx, head_idx=head_idx):
                    def attention_hook(module, input, output):
                        attention_scores = output[1]
                        attention_map = attention_scores[0, head_idx, :, :]
                        self.attentions[(layer_idx, head_idx)] = attention_map.to(self.device)

                    return attention_hook
                layer.self_attn.register_forward_hook(hook_wrapper())
            else:
                print(f"Warning: 'self_attn' attribute not found in model {model}.\n"
                      f"(The error indicates that in model {model}, the name of self-attention is not 'self_attn'. Please check and modify it.)")
                exit()

    def get_attention(self):
        topk_attention = torch.stack([self.attentions[(layer_idx, head_idx)] for layer_idx, head_idx in self.topk_head_index])
        return topk_attention  # [topk, seq, seq]


class Model:
    def __init__(self, args):
        self.top_rank = args.top_rank
        self.model, self.tokenizer = self.load_model_and_tokenizer(args)
        self.device = self.model.device
        self.set_up(args.model, args.topk, args.size)
        self.context_ids = None


    def load_model_and_tokenizer(self, args):
        print(f"Loaded models: {args.model}")
        #model_path = f'../models/{model_name}'
        model_path = args.model_path

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = AutoConfig.from_pretrained(model_path)
        self.vocab_size = config.vocab_size
        self.max_length = min(2048, config.max_position_embeddings)
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads

        return model, tokenizer


    def set_up(self, model_name, topk, size):
        ''' some initial settings '''
        model_name_lower = model_name.lower()

        # end_token_ids:
        # Represents some non-semantic characters that will be output before or after the answer tokens, such as '\n' or <|eot_id|>.
        model_config = {
            'llama-2': {
                'end_token_ids': [13, 29871],  # '\n', ' '
            },
            'mistral': {
                'end_token_ids': [28705],  # ' '
            },
            'llama-3': {
                'end_token_ids': [13, 198, 220, 128009],  # '.', '\n', ' ', '<|eot_id|>'
            }
        # If you want to use DAGCD on models from other families,
        # you need to first determine the corresponding end_token_ids, which are generally the IDs corresponding to '\n' or <|eot_id|>.
        }

        for key, config in model_config.items():
            if key in model_name_lower:
                self.end_token_ids = config['end_token_ids']
                break
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        
        path_prefix = f'../detector/{model_name}'
        self.detector = joblib.load(f'{path_prefix}-top{topk}-size{size}-model.pkl')

        with open(f'{path_prefix}-top{topk}-size{size}-para.pkl', 'rb') as f:
            data = pickle.load(f)
            self.topk_head_weights = torch.tensor(data['weights'], device=self.device).view(-1, 1)
            self.topk_head_index = []
            for index in data['topk_indices']:
                layer_idx = index // self.num_heads
                head_idx = index % self.num_heads
                self.topk_head_index.append([layer_idx, head_idx])
            self.topk_heads = data['topk_indices']
        

        # scaling factor
        self.alpha = 4 if 'chat' in model_name_lower or 'instruct' in model_name_lower else 2

        # Initialize the AttentionExtractor and register hooks
        self.attention_extractor = AttentionExtractor(self.model, self.topk_head_index)
        self.attention_extractor.register_hooks()


    def print_distribution_topk(self, probs, topn):
        ''' Print the information corresponding to the top <topn> tokens in the distribution <probs>. '''
        top_values, top_indices = torch.topk(probs, topn)
        top_values = top_values.squeeze(0)
        top_indices = top_indices.squeeze(0)
        # ids to tokens
        for topk, (val, ids) in enumerate(zip(top_values, top_indices)):
            ids = ids.item()
            token = self.tokenizer.decode([ids])
            logger.info(f'rank{topk + 1},  value:{val}, token:{token}, ids:{ids}')


    def get_context_token_rank(self, original_probs, context_token_ids):
        '''
        return: The ranking of all context tokens in the original distribution.
        '''
        sorted_indices = torch.argsort(original_probs, descending=True)
        ranks = torch.nonzero(sorted_indices == context_token_ids.unsqueeze(1), as_tuple=True)[1] + 1
        return ranks.to(self.device)


    def locate_context_position(self, offset_mapping, prompt, context):
        """
        Locate the start and end token indices of the context string within input_ids based on offset_mapping and the context string in the prompt.
        Parameters:
            offset_mapping: A list or tensor containing the character start and end positions
                            of each token in the prompt, formatted as
                            [(start1, end1), (start2, end2), ...].
            prompt: The original full prompt string.
            context: The substring to be located.

        Returns:
            (start_token_idx, end_token_idx) - The start and end token indices of the context within input_ids.
            If the context cannot be found, returns (None, None).
        """

        # Locate the character position of context in the prompt
        start_char = prompt.find(context)
        if start_char == -1:
            return None, None  # If target string not fuond
        end_char = start_char + len(context)

        token_starts = offset_mapping[:, 0]
        token_ends = offset_mapping[:, 1]

        valid_tokens = (token_ends > start_char) & (token_starts < end_char)
        token_indices = np.nonzero(valid_tokens)[0]

        # return the context starting and ending token indices
        return token_indices[0], token_indices[-1]


    def get_unique_attn(self, raw_attention):
        '''
        In the case of multiple attention values for the same token_ids,
        we use the average to represent the attention of that token.
        '''
        unique_context_ids, inverse_indices = torch.unique(self.context_ids, return_inverse=True)
        unique_context_len = len(unique_context_ids)
        unique_attention = torch.zeros((raw_attention.shape[0], unique_context_len), device=self.device)  # [num_dist, unique_len]

        for idx in range(unique_context_len):
            mask = (inverse_indices == idx)
            avg = raw_attention[:, mask].mean(dim=1)
            unique_attention[:, idx] = avg

        return unique_context_ids, unique_attention


    def get_utilization_distribution(self, topk_attentions, context_start_idx, context_end_idx, token_level_probability_distribution):
        """
        Locate the start and end token indices of the context string within token_ids
        based on offset_mapping and the context string in the prompt.

        Parameters:
            topk_attentions: topk_heads' attention map
            context_start_idx: The starting position of context in input_ids
            context_end_idx: The ending position of context in input_ids
            token_level_probability_distribution: Original distribution

        Returns:
            utilization_distribution
        """

        # step1: Context Utilization Detection
        # 1.1 Feature Data Collection
        sequence_attention = topk_attentions[:, -1, :]  # [topk_head, seq_len]
        context_attention = sequence_attention[:, context_start_idx: context_end_idx + 1]  # [topk_head, context_len]
        unique_context_ids, unique_context_attention = self.get_unique_attn(context_attention)

        # 1.2 Normalize attention across the unique context tokens
        unique_context_attention_sum = unique_context_attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        unique_context_attention_ratio = unique_context_attention / unique_context_attention_sum  # [len(unique_context_ids), topk_head]

        # 1.3 detect utilized-tokens
        data = unique_context_attention_ratio.permute(1, 0).cpu().numpy()
        data = np.nan_to_num(data, nan=0.0)
        mask = torch.tensor(self.detector.predict(data), device=self.device, dtype=torch.float32)

        # 1.4 Top-Rank Constraint
        ranks = self.get_context_token_rank(token_level_probability_distribution, unique_context_ids)
        mask[ranks > self.top_rank] = 0

        # step2: calculate utilization distribution
        utilization_distribution = torch.zeros(self.vocab_size, device=self.device)
        if mask.sum() > 0:
            informative_head = (unique_context_attention_ratio * self.topk_head_weights).sum(dim=0) * mask
            informative_head_sum = informative_head.sum().clamp_min(1e-8)
            informative_head = informative_head / informative_head_sum
            utilization_distribution[unique_context_ids] = informative_head.float()

        logger.info('***** Utilization Distribution *****')
        self.print_distribution_topk(utilization_distribution, topn=10)

        return utilization_distribution.unsqueeze(0)


    def normalized_entropy(self, dist):
        ''' return normalized entropy of distribution <dist> '''
        entropy = -torch.sum(dist * torch.log(dist))
        max_entropy = torch.log(torch.tensor(self.vocab_size, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        return normalized_entropy.item()


    def generate(self,question: str, context: str, gen_max_length: int = 10, use_rag_hallu=True, numk=3):
        '''
        Parameters:
            question: question
            context: context
            gen_max_length: Max new token length
            use_dagcd: Whether to use DAGCD
        Returns:
            generated token ids
        '''
        prompt = f'Given the following information:{context}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:'
        tokenized_inputs = self.tokenizer(prompt,
                                          return_tensors="pt",
                                          return_offsets_mapping=True,
                                          truncation=True,
                                          max_length = self.max_length - gen_max_length
                                          ).to(self.device)
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        offset_mapping = tokenized_inputs['offset_mapping'][0]
        if torch.is_tensor(offset_mapping):
            offset_mapping = offset_mapping.cpu().numpy()
        else:
            offset_mapping = np.array(offset_mapping)

        context_start, context_end = self.locate_context_position(offset_mapping, prompt, context)
        self.context_ids = input_ids.squeeze(0)[context_start: context_end + 1]

        ##################################  Greedy  #######################################
        with torch.no_grad():
            ### 1. Get top k tokens in first generated token position for marginalization 
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, return_dict=True, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            probs = F.softmax(next_token_logits, dim=-1)  # (1, vocab_size)
            topk_probs, topk_tokens = torch.topk(probs, k=numk, dim=-1, sorted=False)  # (1, topk)
            max_index = torch.argmax(topk_probs, dim=-1).item()  # (1,)
            
            ### 2. Generate greedy tokens
            candidate_tokens = []
            candidate_log_probs = []
            candidate_answers = []
            for i in range(numk):
                first_token_id = topk_tokens[0, i].unsqueeze(0).unsqueeze(0)
                log_probs = [torch.log(topk_probs[0, i]).item()]  # log prob of first token
                
                if gen_max_length <= 1:
                    candidate_tokens.append(first_token_id[0])
                    candidate_log_probs.append(log_probs[0])
                    candidate_answers.append(normalize_answer(self.tokenizer.decode(first_token_id[0], skip_special_tokens=True)))
                else:
                    input_ids_i = torch.cat([input_ids, first_token_id], dim=-1)  # (1, input_len + 1)
                    attention_mask_i = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)], dim=-1)
                    
                    outputs = self.model.generate(input_ids=input_ids_i,
                                        attention_mask=attention_mask_i,
                                        max_new_tokens=gen_max_length - 1,
                                        do_sample=False,
                                        num_beams=1,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        output_attentions=True)
                        
                    ## Decode the generated tokens
                    generated_ids = torch.cat([first_token_id, outputs.sequences[:, input_ids_i.shape[-1]:]], dim=-1)  # only new tokens
                    answer_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    norm_ans = normalize_answer(answer_text)
                    
                    ## Get the average log prob
                    for score_dist in outputs.scores:
                        top_token_id = score_dist.argmax(dim=-1).item()
                        log_prob = torch.log(F.softmax(score_dist, dim=-1)[0, top_token_id]).item()
                        log_probs.append(log_prob)
                        
                    avg_log_prob = sum(log_probs) / len(log_probs)
                    
                    candidate_tokens.append(generated_ids[0])
                    candidate_log_probs.append(avg_log_prob)
                    candidate_answers.append(norm_ans)
                
            pred_tokens = candidate_tokens[max_index]
            
            ##################################  RAG Hallu  #######################################
            if use_rag_hallu:
                ### 1. Get pred_ans after sentence
                pred_ans = candidate_answers[max_index]
                norm_pred_ans = normalize_answer(pred_ans)
                new_prompt = f'Given the following information:{context}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer: {norm_pred_ans}\nReason:'
                
                new_tokenized_inputs = self.tokenizer(new_prompt,
                                            return_tensors="pt",
                                            truncation=True,
                                            max_length=self.max_length - gen_max_length
                                            ).to(self.device)
                
                reason_outputs = self.model.generate(input_ids=new_tokenized_inputs['input_ids'],
                                             attention_mask=new_tokenized_inputs['attention_mask'],
                                             max_new_tokens=50,
                                             do_sample=False,
                                             num_beams=1,
                                             return_dict_in_generate=True,
                                             output_attentions=True)
                
                reason_ids = reason_outputs.sequences[:, new_tokenized_inputs['input_ids'].shape[-1]:]
                reason_text = self.tokenizer.decode(reason_ids[0], skip_special_tokens=True)
                
                ### 2. Get avg_log_prob of after sentence for candidates
                #new_candidate_log_probs = [avg_reason_log_prob]
                new_candidate_log_probs = []
                for i, cand_ans in enumerate(candidate_answers):                        
                    
                    alt_prompt = f'Given the following information:{context}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer: {cand_ans}\nExplain why this is the correct answer:'
                    alt_tokenized_inputs = self.tokenizer(alt_prompt,
                                                return_tensors="pt",
                                                truncation=True,
                                                max_length=self.max_length - gen_max_length
                                                ).to(self.device)
                    total_log_prob = 0.0
                    input_ids_alt = alt_tokenized_inputs['input_ids']
                    attention_mask_alt = alt_tokenized_inputs['attention_mask']

                    for j in range(reason_ids.shape[1]):
                        outputs = self.model(input_ids=input_ids_alt,
                                     attention_mask=attention_mask_alt,
                                     return_dict=True,
                                     output_attentions=True)
                        logits = outputs.logits[:, -1, :]
                        prob_dist = F.softmax(logits, dim=-1)
                        tgt_token = reason_ids[0, j].item()
                        total_log_prob += torch.log(prob_dist[0, tgt_token]).item()

                        input_ids_alt = torch.cat([input_ids_alt, reason_ids[:, j:j+1]], dim=-1)
                        attention_mask_alt = torch.cat([attention_mask_alt, torch.ones((1, 1), device=self.device)], dim=-1)

                    avg_alt_log_prob = total_log_prob / reason_ids.shape[1]
                    new_candidate_log_probs.append(avg_alt_log_prob)

                # Marginalization
                print(candidate_log_probs)
                print(new_candidate_log_probs)
                candidate_log_probs = torch.tensor(candidate_log_probs)
                new_candidate_log_probs = torch.tensor(new_candidate_log_probs)
                joint_log_probs = candidate_log_probs + new_candidate_log_probs
                joint_probs = torch.exp(joint_log_probs)
                bi_dir_prob = joint_probs[max_index] / joint_probs.sum()
                
                ##################################  RAG Hallu  #######################################

            if use_rag_hallu:
                return pred_tokens, reason_text, max_index, topk_probs.squeeze().tolist(), (topk_probs[0, max_index].item(), torch.exp(candidate_log_probs[max_index]), bi_dir_prob), (torch.exp(candidate_log_probs).tolist(), torch.exp(new_candidate_log_probs).tolist()), candidate_answers
            else:
                return pred_tokens


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--data', type=str, default='HotpotQA')
    parser.add_argument('--topk', type=int, default=10, help='top-k features')
    parser.add_argument('--top_rank', type=int, default=10, help='top rank constraint')
    parser.add_argument('--size', type=int, default=2000, help='LR-train-size')
    parser.add_argument('--numk', type=int, default=3, help='number of marialized tokens')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--ratio', type=float, default=0.02, help='sample ratio for validation data')
    args = parser.parse_args()
    args.model_path = args.model
    args.model = os.path.basename(args.model).lower()

    set_seed(args.seed)
    logger = set_logger(args)

    model = Model(args)
    data_path = f'../datasets/validation/{args.data}.parquet'
    datas = utils.load_parquet_data(data_path)
    
    # Sample 10%
    sample_size = int(len(datas) * args.ratio)
    datas = random.sample(datas, sample_size)

    results = []

    greedy = True  # Whether to generate greedy answers

    for i, data in enumerate(tqdm(datas)):
        question = data['question']
        context = data['context']
        gold_ans = data['answers']

        tgt_len = max(len(model.tokenizer.encode(ans, add_special_tokens=False)) for ans in gold_ans)

        # Greedy answer
        if greedy:
            greedy_tokens = model.generate(
                question=question,
                context=context,
                gen_max_length=tgt_len,
                use_rag_hallu=False
            )
            if greedy_tokens is None: continue
            greedy_ans = model.tokenizer.decode(greedy_tokens, skip_special_tokens=True)

        # RAG Hallu answers
        rag_hallu_tokens, sentence, max_index, topk_probs, (first_prob, before_prob, after_prob), (before_prob_list, after_prob_list), answers = model.generate(
            question=question,
            context=context,
            gen_max_length=tgt_len,
            use_rag_hallu=True,
            numk=args.numk
        )
        if rag_hallu_tokens is None: continue
        pred_ans = model.tokenizer.decode(rag_hallu_tokens, skip_special_tokens=True)

        if greedy:
            res = {'context': context, 'question': question, 'gold_ans': gold_ans, 'greedy_ans': greedy_ans, 'pred_ans': pred_ans, "max_index": max_index, "answer_list": ', '.join(answers), "generated_sentence": sentence, "topk_probs": ' '.join([str(prob) for prob in topk_probs]), 'answer_prob': float(before_prob), 'marginalized_prob': float(after_prob), 'answer_prob_list': ' '.join([str(prob) for prob in before_prob_list]), 'marginalized_prob_list': ' '.join([str(prob) for prob in after_prob_list])}
            logger.info(f"Q:{question}\n"
                        f"Golden Answer: {gold_ans}\n"
                        f"Greedy Answer: {greedy_ans}\n"
                        f"RAG Hallu Answer: {pred_ans}\n"
                        f"Generated Sentence: {sentence}\n"
                        f"Answer Prob: {before_prob}\n"
                        f"Marginalized Prob: {after_prob}\n"
                        f"Answer Prob List: {before_prob_list}\n"
                        f"Marginalized Prob List: {after_prob_list}")
        else:
            res = {'context': context, 'question': question, 'gold_ans': gold_ans, 'pred_ans': pred_ans, "max_index": max_index, "answer_list": ', '.join(answers), "generated_sentence": sentence, "topk_probs": ' '.join([str(prob) for prob in topk_probs]), 'answer_prob': float(before_prob), 'marginalized_prob': float(after_prob), 'answer_prob_list': ' '.join([str(prob) for prob in before_prob_list]), 'marginalized_prob_list': ' '.join([str(prob) for prob in after_prob_list])}
            logger.info(f"Q:{question}\n"
                        f"Golden Answer: {gold_ans}\n"
                        f"RAG Hallu Answer: {pred_ans}\n"
                        f"Generated Sentence: {sentence}\n"
                        f"Answer Prob: {before_prob}\n"
                        f"Marginalized Prob: {after_prob}\n"
                        f"Answer Prob List: {before_prob_list}\n"
                        f"Marginalized Prob List: {after_prob_list}")

        logger.info('*-'*60)
        logger.info('\n\n')

        results.append(res)

    save_path = f'../results/{args.model}/{args.data}-{args.size}-{int(100*args.ratio)}per-seed{args.seed}-gen.jsonl'
    utils.save_as_parquet(save_path, results)


    ######################################### evaluate #########################################
    # filter token length > 2048 (because truncation)
    new_data = []
    for d in results:
        prompt = f'Given the following information:{d["context"]}\nAnswer the following question based on the given information with one or few words: {d["question"]}\nAnswer:'
        if len(model.tokenizer.encode(prompt)) <= 2048: new_data.append(d)
    results = new_data


    ############# greedy results #############
    if greedy:
        greedy_f1, _, _ = utils.evaluate_f1(results, gold_ans='gold_ans', pred_ans='greedy_ans')
        greedy_em, _, _ = utils.evaluate_em(results, gold_ans='gold_ans', pred_ans='greedy_ans')

        logger.info('*' * 30)
        logger.info('----- Greedy performance -----')
        logger.info(f'EM: {greedy_em * 100}%')
        logger.info(f'F1: {greedy_f1 * 100}%\n\n')


    ############# dagcd results #############
    dagcd_f1, _, _ = utils.evaluate_f1(results, gold_ans='gold_ans', pred_ans='pred_ans')
    dagcd_em, _, _ = utils.evaluate_em(results, gold_ans='gold_ans', pred_ans='pred_ans')
    logger.info('----- RAG Hallu performance -----')
    logger.info(f'EM: {dagcd_em * 100}%')
    logger.info(f'F1: {dagcd_f1 * 100}%')
    logger.info('*' * 30)

    print('----- RAG Hallu performance -----')
    print(f'EM: {dagcd_em * 100}%')
    print(f'F1: {dagcd_f1 * 100}%')

    with open(f'../results/{args.model}/{args.data}-top{args.topk}-size{args.size}.txt', 'w') as file:
        if greedy:
            file.write('----- Greedy performance -----\n')
            file.write(f'EM: {greedy_em * 100}%\n')
            file.write(f'F1: {greedy_f1 * 100}%\n\n')

        file.write('----- RAG Hallu performance -----\n')
        file.write(f'EM: {dagcd_em * 100}%\n')
        file.write(f'F1: {dagcd_f1 * 100}%')