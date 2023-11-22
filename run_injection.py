import os
import json
import numpy as np
import random
from tqdm import tqdm, trange
import argparse
import spacy
nlp = spacy.load("en_core_web_sm")

from qa_utils import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='NaturalQuestions', choices=['TriviaQA', 'SQuAD', 'NaturalQuestions', 'HotpotQA']) #
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev']) #
    parser.add_argument('--n_samples', type=int, default=500) #

    # arguments for injection
    parser.add_argument('--position', type=str, default='end', choices=['start', 'middle', 'end', 'random']) #
    parser.add_argument('--attack_type', type=str, default='direct', choices=['direct', 'ignore_previous', 'ignore_next', 'ignore_both', 
                                                                              'primary_prefix', 'primary_suffix', 'first_prefix', 'first_suffix',
                                                                              'order_prefix']) #

    parser.add_argument('--task_type', type=str, default='irrelevant', choices=['irrelevant', 'relevant']) #
    parser.add_argument("--test_mode", type=str, required=True, choices=['original', 'injected', 'relevant'], help="whether to inject tasks.")

    args, unknown = parser.parse_known_args()

    # load attack injections
    with open("./prompts/ignore_prefix.json", 'r') as file:
        ignore_prefixes = json.load(file)
    with open("./prompts/ignore_suffix.json", 'r') as file:
        ignore_suffixes = json.load(file)
    with open("./prompts/answer_primary.json", 'r') as file:
        answer_primary = json.load(file)
    with open("./prompts/answer_first.json", 'r') as file:
        answer_first = json.load(file)
    with open("./prompts/answer_order.json", 'r') as file:
        answer_order = json.load(file)

    # load self-instruct instructions (irrelevant task instructions)
    instructions = []
    instruction_path = f"./data/qa/self-instruct/gpt3_filtered_instances_82K.jsonl"
    with open(instruction_path, 'r') as f:
        for line in f:
            instruction = json.loads(line)
            if instruction['valid']:
                instructions.append(instruction['instruction'])
    print(f"Number of irrelevant tasks: {len(instructions)}")
    
    # load relevant qa (relevant task instructions)
    relevant_data_path = f"./data/qa/{args.dataset}/{args.split}-{args.n_samples}-aug.json" 
    with open(relevant_data_path, "r") as f:
        examples = json.load(f)
    num_examples = len(examples)
    assert num_examples == args.n_samples
    print(f"Number of relevant questions: {num_examples}")
            
    # construct test data
    if args.test_mode == "injected":
        eval_data_path = f"./inputs/{args.test_mode}-{args.task_type}-{args.position}-{args.attack_type}-{args.dataset}-{args.split}-{args.n_samples}.json"    
    else: # original/relevant
        eval_data_path = f"./inputs/{args.test_mode}-{args.dataset}-{args.split}-{args.n_samples}.json" 
    
    position = args.position
    task_type = args.task_type
    attack_type = args.attack_type

    if os.path.exists(eval_data_path):
        with open(eval_data_path, "r") as file:
            eval_data = json.load(file)
    else:
        eval_data = []
    num_examples = len(eval_data)
    print(f"Number of existing samples: {num_examples}")

    for idx in trange(num_examples, len(examples)):

        example = examples[idx]
        context = example['context']

        qas = example['qas'][0]
        question = qas['question']
        qid = qas['qid'] if "qid" in qas else qas['id']
        answers = qas['answers']

        relevant_qas = random.choice(example['qas'][1:])
        relevant_question = relevant_qas['question']
        relevant_qid = relevant_qas['qid'] if "qid" in relevant_qas else relevant_qas['id']
        relevant_answers = relevant_qas['answers']

        if args.test_mode == "injected":
            # select injection task
            if args.task_type == "irrelevant":
                injected_question = random.choice(instructions)
                injected_answers = [""]
            elif args.task_type == "relevant":
                injected_question = relevant_question
                injected_answers = relevant_answers

            # select injection prefix
            injection = injected_question
            injected_prefix = ""
            injected_suffix = ""
            INJECTION_TEMPLATE = "{prefix} Question: {question} {suffix}"
            if args.attack_type == "ignore_previous":
                injected_prefix = random.choice(ignore_prefixes)
            elif args.attack_type == "ignore_next":
                injected_suffix = random.choice(ignore_suffixes)
            elif args.attack_type == "ignore_both":
                injected_prefix = random.choice(ignore_prefixes)
                injected_suffix = random.choice(ignore_suffixes)
            elif args.attack_type == "first_prefix":
                injected_prefix = random.choice(answer_first)
            elif args.attack_type == "first_suffix":
                injected_suffix = random.choice(answer_first)
            elif args.attack_type == "order_prefix":
                injected_prefix = random.choice(answer_order)
            elif args.attack_type == "primary_prefix":
                INJECTION_TEMPLATE = "{prefix} Primary Question: {question} {suffix}"
                injected_prefix = random.choice(answer_primary)
            elif args.attack_type == "primary_suffix":
                INJECTION_TEMPLATE = "{prefix} Primary Question: {question} {suffix}"
                injected_suffix = random.choice(answer_primary)
            # fill in the blank
            injection = INJECTION_TEMPLATE.format(prefix=injected_prefix, 
                                                    question=injected_question,
                                                    suffix=injected_suffix)
            
            # construct test samples
            if args.position == "start":
                if context.startswith("<P>"):
                    context_sentences = context.split("<P>")
                    context_sentences.insert(1, injection)
                    context_sentences.insert(1, "<P>")
                else:
                    context_sentences = [injection, context]
            elif args.position == "end":
                if context.endswith("</P>"):
                    context_sentences = context.split("</P>")
                    context_sentences.append(injection)
                    context_sentences.append("</P>")
                else:
                    context_sentences = [context, injection]
            elif args.position == "middle":
                context_sentences = [str(s) for s in nlp(context).sents]
                context_sentences.insert(int(len(context_sentences)/2), injection)
            elif args.position == "random":
                context_sentences = [str(s) for s in nlp(context).sents]
                context_sentences.insert(random.randint(0,len(context_sentences)-1), injection)
            injected_context = " ".join(context_sentences)

        elif args.test_mode == "original":
            injected_prefix = ""
            injected_question = ""
            injected_answers = [""]
            injection = ""
            attack_type = ""
            task_type = ""
            position = ""
            injected_context = context # no injection

        elif args.test_mode == "relevant":
            question = relevant_question # evaluate the injected question (as original question)
            answers = relevant_answers
            qid = ""
            injected_prefix = ""
            injected_question = ""
            injected_answers = [""]
            injection = ""
            attack_type = ""
            task_type = ""
            position = ""
            injected_context = context # no injection

        test = {
                "context": context,
                "question": question,
                "answers": answers,
                "qid": qid,
                "injection": injection,
                "injected_context": injected_context,
                "injected_question": injected_question,
                "injected_answers": injected_answers,
                "attack_type": attack_type,
                "task_type": task_type,
                "position": position,
            }
        eval_data.append(test)
    
    assert len(eval_data) == len(examples)
    with open(eval_data_path, "w", encoding='utf-8') as file:
        json.dump(eval_data, file, indent=4, ensure_ascii=False)
