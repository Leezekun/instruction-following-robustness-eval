import re
import json
import string
import argparse
from collections import Counter
import torch
from evaluate import load
bertscore = load("bertscore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))    

def recall_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return (ground_truth in prediction)

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def bertscore_score(prediction, ground_truth):
    with torch.inference_mode():
        score = bertscore.compute(predictions=[normalize_answer(prediction)], references=[normalize_answer(ground_truth)], lang='en')['f1'][0]
    return score

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate_qa(prediction, ground_truths, skip_no_answer=False):
    def has_answer(ground_truths):
        for g in ground_truths:
            if g: return True
        return False
    
    if prediction and has_answer(ground_truths):
        exact_match = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        btscore = metric_max_over_ground_truths(
            bertscore_score, prediction, ground_truths)
    else:
        exact_match, f1, btscore = 0., 0., 0.

    return {'exact_match': exact_match*100.0, 'f1': f1*100.0, 'bertscore': btscore}
