import re
from collections import defaultdict

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm


def preprocess_gqa(responses):
    for entry in responses:
        response = entry["answer"]
        response = response.strip().split(".")[0].split(",")[0].split("!")[0].lower()
        if "is " in response:
            response = response.split("is ")[1]
        if "are " in response:
            response = response.split("are ")[1]
        if "a " in response:
            response = response.split("a ")[1]
        if "an " in response:
            response = response.split("an ")[1]
        if "the " in response:
            response = response.split("the ")[1]
        if " of" in response:
            response = response.split(" of")[0]
        response = response.strip()
        entry["answer"] = response
    return responses


def eval_vqa(responses):
    scores = {"correct": 0, "all": 0}
    for res in responses:
        pred = res["prediction"].lower()
        if isinstance(res["answer"], list):
            answers = set([a.lower() for a in res["answer"]])
        else:
            answers = set([res["answer"].lower()])
        if pred in answers:
            scores["correct"] += 1
        scores["all"] += 1
    scores["accuracy"] = float(scores["correct"]) / float(scores["all"])
    print(f"Accuracy: {scores['accuracy']}")
    return scores


def eval_caption(responses):
    gts = {}  # Ground Truths
    res = {}  # Results
    for item in responses:
        q_id = item["question_id"]
        res[q_id] = [item["prediction"]]
        gts[q_id] = item["answer"]
    scores = {}
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    scores["cider"] = cider_score
    print(f"CIDEr Score: {cider_score:.4f}")
    bleu_scorer = Bleu()
    bleu_score, _ = bleu_scorer.compute_score(gts, res)
    scores["bleu"] = bleu_score
    print(f"BLEU Score: {bleu_score}")
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    print(f"Rouge Score: {rouge_score:.4f}")
    scores["rouge"] = rouge_score
    return scores


def eval_results(task: str, responses):
    if task == "gqa":
        responses = preprocess_gqa(responses)
        return eval_vqa(responses)
    elif task == "coco_caption":
        return eval_caption(responses)
    else:
        raise NotImplementedError()
