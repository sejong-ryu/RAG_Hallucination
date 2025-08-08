import utils

if __name__ == '__main__':
    data = utils.load_parquet_data('../results/llama-2-7b-hf/HotpotQA-2000.jsonl')

    # evaluate Greedy
    print('------------------Greedy------------------')
    greedy_em, _, _ = utils.evaluate_em(data, 'gold_ans', 'greedy_ans')
    greedy_f1, _, _ = utils.evaluate_f1(data, 'gold_ans', 'greedy_ans')

    print(f"EM: {greedy_em * 100}")
    print(f"F1: {greedy_f1 * 100}\n")

    # evaluate DAGCD
    print('------------------RAG Hallu------------------')
    ours_em, _, _ = utils.evaluate_em(data, 'gold_ans', 'pred_ans')
    ours_f1, _, _ = utils.evaluate_f1(data, 'gold_ans', 'pred_ans')


    print(f"EM: {ours_em * 100}")
    print(f"F1: {ours_f1 * 100}\n")