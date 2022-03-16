from typing import Any, Tuple, Dict
import numpy as np
from datasets import Metric, Sequence
from transformers import PreTrainedTokenizer


def metric_result_to_dict(metric_name: str, metric_output: Any):
    if metric_name == "sacrebleu":
        return {f"BLEU-{i+1}": v for i, v in enumerate(metric_output["precisions"])} | {
            "Bleu-Score": metric_output["score"]
        }
    elif metric_name == "rouge":
        return {k: v.mid.fmeasure for k, v in metric_output.items()}
    elif metric_name == "meteor":
        return {k: v for k, v in metric_output.items()}
    elif metric_name == "bertscore":
        return {f"BertScore-{k}": sum(v) / len(v) for k, v in metric_output.items() if k != "hashcode"}
    else:
        return ValueError(f"{metric_name} metric not recognized.")


def compute_metrics(eval_preds: Tuple, tokenizer: PreTrainedTokenizer, validation_metrics: Dict[str, Metric]):
    preds, labels = eval_preds
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = {}
    for k, metric in validation_metrics.items():
        # Some metrics (e.g., BLEU) require references as list of lists since there could be multiple valid
        # references for a single prediction. This is not our case so, in that case, we create a list of 1-element lists
        references = [[l] for l in labels] if isinstance(metric.features["references"], Sequence) else labels
        metric_result = metric.compute(predictions=preds, references=references)
        result = result | metric_result_to_dict(k, metric_result)
    return result
