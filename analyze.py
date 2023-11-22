import glob
import json
from collections import Counter
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

for dataset in ["NaturalQuestions", "TriviaQA", "SQuAD", "HotpotQA"]:
    data_path = f"./inputs/injected-relevant-end-direct-{dataset}-dev-1000.json"
    with open(data_path, "r") as file:
        data = json.load(file)

    context_lens, original_q_lens, original_a_lens, injected_q_lens, injected_a_lens = [], [], [], [], []
    for dp in data:
        context = dp["context"]
        orig_q = dp["question"]
        orig_a = dp["answers"][0]
        injected_q = dp["injected_question"]
        injected_a = dp["injected_answers"][0]

        context_len = len(tokenizer(context).input_ids)
        orig_q_len = len(tokenizer(orig_q).input_ids)
        orig_a_len = len(tokenizer(orig_a).input_ids)
        injected_q_len = len(tokenizer(injected_q).input_ids)
        injected_a_len = len(tokenizer(injected_a).input_ids)

        context_lens.append(context_len)
        original_q_lens.append(orig_q_len)
        original_a_lens.append(orig_a_len)
        injected_q_lens.append(injected_q_len)
        injected_a_lens.append(injected_a_len)

    context_lens = np.array(context_lens)
    original_q_lens = np.array(original_q_lens)
    original_a_lens = np.array(original_a_lens)
    injected_q_lens = np.array(injected_q_lens)
    injected_a_lens = np.array(injected_a_lens)
    
    print(dataset)
    print(np.mean(context_lens))
    print(np.mean(original_q_lens))
    print(np.mean(original_a_lens))
    print(np.mean(injected_q_lens))
    print(np.mean(injected_a_lens))
    print(np.max(context_lens))
    print(np.max(original_q_lens))
    print(np.max(original_a_lens))
    print(np.max(injected_q_lens))
    print(np.max(injected_a_lens))
    print()