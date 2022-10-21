import argparse
from typing import List, Dict

from scipy.stats import pearsonr
from tqdm import tqdm
import random

from io_utils import load_model_outputs, ModelOutput
import numpy as np
import json
import math
import os
import shutil

ACTUAL_SCORE_TH = 0.5

def get_confusion_matrix(actual_scores: List[float], predicted_scores: List[float], threshold: float, classifier_type: str):
    """
    Computes the accuracy of a model given actual scores, its predictions, and a classification threshold.
    :param actual_scores: the actual label is considered positive (label=1) if the actual score is above 0.5
    :param predicted_scores: the predicted label is considered positive (label=1) if the predicted score is above the given threshold
    :param threshold: the threshold for computing predicted labels
    :return: the accuracy of the predictions
    """
    assert len(actual_scores) == len(predicted_scores)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for actual_score, predicted_score in zip(actual_scores, predicted_scores):
        if actual_score is None:
            continue
        if classifier_type == 'discriminative':
            predicted_score_cmp = predicted_score[0]
        else:
            predicted_score_cmp = predicted_score[0] - predicted_score[1]

        if actual_score >= ACTUAL_SCORE_TH:
            if predicted_score_cmp >= threshold:
                TP += 1
            else:
                FN += 1
        else:
            if predicted_score_cmp >= threshold:
                FP += 1
            else:
                TN += 1
    return TP,FP,FN,TN

def evaluate_dev(all_dev_examples, attribute_name, classifier_type='discriminative', possible_th=None):
    actual_scores = [example['labels'][attribute_name] for example in all_dev_examples]
    predicted_scores = [example['scores'][attribute_name] for example in all_dev_examples]


    best_acc, best_threshold, best_eval = -1, 0, -1
    for threshold in possible_th:
        TP,FP,FN,TN = get_confusion_matrix(actual_scores, predicted_scores, threshold, classifier_type)
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        recall = TP/(TP+FN) if TP+FN != 0 else 0
        precision = TP/(TP+FP) if TP+FP != 0 else 0
        F1 = 2*recall*precision/(recall+precision) if recall+precision != 0 else 0
        eval = F1
        if eval > best_eval:
            best_eval = eval
            best_threshold = threshold
            best_acc = accuracy
    print(f'The optimal threshold is {best_threshold}, with F1 of {best_eval}, with accuracy of {best_acc}')

    num_toxicity = 0
    num_all = 0
    for s in actual_scores:
        if s is None:
            continue
        if s>=ACTUAL_SCORE_TH:
            num_toxicity += 1
        num_all += 1
    print(f'num_toxicity is {num_toxicity}, num_safe is {num_all-num_toxicity}, num_all is {num_all}')

    return best_threshold, best_eval

def evaluate_test(all_test_examples, attribute_name, best_threshold, classifier_type='discriminative'):
    test_actual_scores = [example['labels'][attribute_name] for example in all_test_examples]
    test_predicted_scores = [example['scores'][attribute_name] for example in all_test_examples]

    if classifier_type == 'discriminative':
        predict_scores = [score[0] for score in test_predicted_scores]
        predict_scores_prior = predict_scores
    else:
        predict_scores = [math.exp(score[0])/(math.exp(score[0])+math.exp(score[1])) for score in test_predicted_scores]
        predict_scores_prior = [math.exp(score[0])/(math.exp(score[0])+math.exp(score[1])*math.exp(best_threshold)) for score in test_predicted_scores]
    corr, _ = pearsonr(test_actual_scores, predict_scores)
    corr_prior, _ = pearsonr(test_actual_scores, predict_scores_prior)


    TP,FP,FN,TN = get_confusion_matrix(test_actual_scores, test_predicted_scores, best_threshold, classifier_type)
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    recall = TP/(TP+FN) if TP+FN != 0 else 0
    precision = TP/(TP+FP) if TP+FP != 0 else 0
    F1 = 2*recall*precision/(recall+precision) if recall+precision != 0 else 0
    evaluate_result = {'pearsonr': corr, 'pearsonr-with-prior': corr_prior, 'accuracy': accuracy, 'recall':recall, 'precision':precision, 'F1':F1, 'TP': TP, 'FP':FP, 'FN':FN, 'TN':TN}
    return evaluate_result

def get_examples(examples, attribute):
    # we estimate the ideal threshold using the dev examples
    examples = [example for example in examples if (example['labels'][attribute] is not None 
                and not math.isnan(example['scores'][attribute][0]) 
                and not math.isnan(example['scores'][attribute][1]))]

    dev_examples_per_class = int(len(examples)/10) #int(len(examples)/2) #7500
    test_examples_per_class = int(len(examples)/10*9) #int(len(examples)/2) #7500

    examples = sorted(examples, key=lambda example: example['labels'][attribute])
    total_examples_per_class = dev_examples_per_class + test_examples_per_class
    bottom_examples = examples[:total_examples_per_class]
    top_examples = examples[-total_examples_per_class:]

    #sum_bottom = sum(ex['labels'][attribute] for ex in bottom_examples) / total_examples_per_class
    #print(f'Average "{attribute}" for bottom examples: '
    #    f'{sum_bottom}')
    #sum_top = sum(ex['labels'][attribute] for ex in top_examples) / total_examples_per_class
    #print(f'Average "{attribute}" for top examples: '
    #    f'{sum_top}')

    rng = random.Random(42)
    rng.shuffle(bottom_examples)
    rng.shuffle(top_examples)

    all_dev_examples = bottom_examples[:dev_examples_per_class] + top_examples[:dev_examples_per_class]
    all_test_examples = bottom_examples[dev_examples_per_class:] + top_examples[dev_examples_per_class:]

    return all_dev_examples, all_test_examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_dir", type=str, required=True,
                        help="Path to a jsonl file containing the texts to be diagnosed, in the format used by RealToxicityPrompts")
    parser.add_argument("--examples_filename", type=str, required=True,
                        help="Path to a jsonl file containing the texts to be diagnosed, in the format used by RealToxicityPrompts")
    parser.add_argument("--attribute", type=str, required=False, default="toxicity",
                        help="")
    parser.add_argument("--delete", type=bool, required=False, default=False,
                        help="")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    checkpoints = [ckpt for ckpt in os.listdir(args.examples_dir) if "checkpoint" in ckpt]
    checkpoints = sorted(checkpoints)
    print(checkpoints)

    Evaluate = {}
    if 'self_detection_dis' in args.examples_filename:
        classifier_type = 'discriminative'
        possible_th = np.linspace(0.05,1,20)
    elif 'self_detection_gen' in args.examples_filename:
        classifier_type = 'generative'
        possible_th = np.linspace(-10,10,40)
        #possible_th = [0]
    else:
        raise NotImplemented

    #dev
    TH = []
    DEV_RESUT = []
    for ckpt in checkpoints:
        examples = []
        with open(os.path.join(os.path.join(args.examples_dir,ckpt), args.examples_filename), 'r', encoding='utf8') as fh:
            for line in fh:
                line_json = json.loads(line)
                examples.append(line_json)

        all_dev_examples, _ = get_examples(examples, args.attribute)
        
        print("ckpt:{}".format(ckpt))
        threshold, eval = evaluate_dev(
            all_dev_examples, attribute_name=args.attribute,
            classifier_type = classifier_type,
            possible_th=possible_th,
        )
        TH.append(threshold)
        DEV_RESUT.append(eval)


    #test
    best_result = max(DEV_RESUT)
    best_ckpt = checkpoints[DEV_RESUT.index(best_result)]
    best_threshold = TH[DEV_RESUT.index(best_result)]

    examples = []
    with open(os.path.join(os.path.join(args.examples_dir,best_ckpt), args.examples_filename), 'r', encoding='utf8') as fh:
        for line in fh:
            line_json = json.loads(line)
            examples.append(line_json)

    _, all_test_examples = get_examples(examples, args.attribute)
    test_result = evaluate_test(
            all_test_examples, attribute_name=args.attribute,
            classifier_type = classifier_type,
            best_threshold=best_threshold,
        )
    test_result['best_eval_result'] = best_result
    test_result['best_ckpt'] = best_ckpt
    test_result['best_threshold'] = best_threshold
    Evaluate[args.attribute] = test_result
    print(test_result)

    #delete checkpoints except the best one
    if args.delete:
        files = [ckpt for ckpt in os.listdir(args.examples_dir) if "checkpoint" in ckpt]

        for f in os.listdir(args.examples_dir):
            if (best_ckpt in f) or ("trainer_state.json" in f):
                continue
            else:
                path = os.path.join(args.examples_dir, f)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)
    
    #save result
    output_file_name = "{}/{}_{}_diagnosis.jsonl".format(
        args.examples_dir,
        args.examples_filename.split('.')[0],
        args.examples_dir.split('/')[-1])

    output_file = open(output_file_name, 'w')
    json.dump(Evaluate, output_file)
    output_file.close()
    print("save to {}".format(output_file_name))