"""
Data from https://github.com/mrqa/MRQA-Shared-Task-2019
"""

import os
import re
import json
import random
import argparse
from tqdm import tqdm, trange
from qa_utils import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='NaturalQuestions', choices=['TriviaQA', 'SQuAD', 'NaturalQuestions', 'HotpotQA']) #
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev']) #
    parser.add_argument('--n_samples', type=int, default=1000) #
    parser.add_argument('--human_eval_num', type=int, default=50) #
    parser.add_argument('--human_eval_group', type=int, default=5) #

    # arguments for generation
    parser.add_argument('--defense', type=str2bool, default=True, help='whether to use instructional prevention for defense') #
    parser.add_argument('--attack_type', type=str, default='direct') #
    parser.add_argument('--position', type=str, default='end', choices=['start', 'middle', 'end', 'random']) #
    parser.add_argument('--task_type', type=str, default='irrelevant', choices=['irrelevant', 'relevant']) #
    parser.add_argument("--test_mode", type=str, required=True, help="whether to inject tasks.")
    parser.add_argument('--n_shot', type=int, default=4, choices=[0,1,2,3,4,5]) #
    parser.add_argument('--template', type=str, default='QCA', choices=['QCA', 'CQA']) #
    
    args, unknown = parser.parse_known_args()

    human_eval_dir = "./human_evals/"
    model_list = ['gpt-3.5-1106', 'claude-2', 'llama-2-70b-chat', 'vicuna-33b-v1.3', 'vicuna-13b-v1.3', 'llama-2-13b-chat', 'alpaca-7b', 'zephyr-7b-beta']
    assert args.human_eval_num % args.human_eval_group == 0

    """
    Step 1: select human evaluation samples
    """
    group_ids_path = human_eval_dir + f"human_eval_{args.human_eval_num}_group{args.human_eval_group}_{args.dataset}-{args.n_samples}.json"
    if os.path.exists(group_ids_path):
        with open(group_ids_path, 'r') as file:
            group_ids = json.load(file)
    else:
        group_ids = []
        indices = list(range(args.n_samples))
        selected_indices = random.sample(indices, args.human_eval_num)
        per_group_num = int(args.human_eval_num//args.human_eval_group)
        for i in range(args.human_eval_group):
            group_data = []
            group_indices = selected_indices[i*per_group_num: (i+1)*per_group_num]
            for ind in group_indices:
                shuffled_model_list = model_list.copy()
                random.shuffle(shuffled_model_list)
                group_data.append({
                    "index": ind,
                    "models": shuffled_model_list
                })
            group_ids.append(group_data)
        with open(group_ids_path, 'w') as file:
            json.dump(group_ids, file, indent=4, ensure_ascii=False)
    
    """
    Step 2: select the corresponding generation
    """
    input_dir = "./inputs/"
    output_dir = "./outputs/"
    if args.test_mode == "injected":
        data_name = f"{args.test_mode}-{args.task_type}-{args.position}-{args.attack_type}-{args.dataset}-{args.split}-{args.n_samples}.json"  
    else:
        data_name = f"{args.test_mode}-{args.dataset}-{args.split}-{args.n_samples}.json" 
    
    model_eval_results = {}
    for model in model_list:
        eval_result_path = output_dir + f"defense{args.defense}-{args.template}-{model}-{args.n_shot}shot-{data_name}"      
        with open(eval_result_path, "r") as file:
            eval_results = json.load(file)
        model_eval_results[model] = eval_results

    for gid in range(args.human_eval_group):
        human_eval_path = human_eval_dir + f"group{gid}-{args.human_eval_num}-defense{args.defense}-{args.template}-{args.n_shot}shot-{data_name}"    
        if not os.path.exists(human_eval_path):
            group_data = []
            for datapoint in group_ids[gid]:
                index = datapoint["index"]
                models = datapoint["models"]
                uniq_context = []
                responses = []
                # iterate over all models
                for model in models:
                    generation = model_eval_results[model][index]
                    context = generation["context"]
                    if context not in uniq_context:
                        uniq_context.append(context)
                    responses.append({
                        "response": generation["generation"],
                        "option": ""
                    })

                assert len(uniq_context) == 1
                assert len(responses) == len(model_list)
                
                group_data.append(
                    {
                    "context": context,
                    "question 1": generation["question"],
                    "answer 1": generation["answers"],
                    "question 2": generation["injected_question"],
                    "answer 2": generation["injected_answers"],
                    "responses": responses
                    }
                )
            with open(human_eval_path, "w") as file:
                json.dump(group_data, file, indent=4, ensure_ascii=False)
                

