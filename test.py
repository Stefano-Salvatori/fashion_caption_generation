from json import decoder
import h5py
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_metric
from transformers import (
    GPT2Model,
    BertModel,
    GPT2TokenizerFast,
    ViTModel,
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
)
from modules.dataset import FashionGenTorchDataset
from modules.fashiongen_utils import DEFAULT_STRINGS_ENCODING, FashionGenDataset
from typing import Literal
import os

# TEST CONFIG
loss_type = "triplet"  # [entropy,triplet]  loss del modello usato per generare le caption
step = 12  # [5..12]
regenerate_predictions = True  # [True, False]  rigenera le predizioni sul validation set o usa quelle salvate su file
regenerate_metrics = True  # [True, False]  rigenera le metriche sul validation set o usa quelle salvate su file
print_metrics = True  # [True, False]

# ## Device & Batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 48

decoder_max_text_length = 64
decoder_min_text_length = 48

# MAIN PATHS
drive_path = "./"  #'drive/MyDrive/Tesi/'
validation_dataset_path = "/home/salvatori/datasets/FashionGen/fashiongen_validation.h5"
checkpoints_path = "./checkpoints/"  # drive_path + 'checkpoints/'
checkpoint_name = "checkpoint-130248"

checkpoint = os.path.join(checkpoints_path, checkpoint_name)
# checkpoints_path + loss_type + '-checkpoint-' + str(step)

# class to hold components of Encoder-Decoder model
class modelComponents:
    def __init__(self, encoder, encoder_checkpoint, decoder, decoder_checkpoint, img_processor, tokenizer):
        self.encoder_checkpoint = encoder_checkpoint
        self.decoder_checkpoint = decoder_checkpoint
        self.img_processor = img_processor
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder


# component configurations
vit_gpt2 = modelComponents(
    encoder=ViTModel,
    encoder_checkpoint="google/vit-base-patch16-224-in21k",
    decoder=GPT2Model,
    decoder_checkpoint="gpt2",
    img_processor=ViTFeatureExtractor,
    tokenizer=GPT2TokenizerFast,
)


# ## Model and Dataset init
def load_data(
    dataset_path: str,
    tokenizer,
    img_processor,
    n_val,
    device: torch.device,
    subcategory: bool = False,
    max_text_length: int = 64,
):
    # encode captions
    cap_val = list()
    for p in tqdm(FashionGenDataset(dataset_path).raw_h5()["input_description"], position=0, leave=True):
        cap_val.append(p[0].decode(DEFAULT_STRINGS_ENCODING))  # DEFUALT_STRINGS_ENCODING = DEFAULT_STRINGS_ENCODING)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    cap_val = list(
        map(
            lambda caption: tokenizer.encode(
                caption.replace("<br>", " "),
                return_tensors="pt",
                max_length=max_text_length,
                padding="max_length",
                truncation=True,
            )[0],
            cap_val,
        )
    )
    data_val = FashionGenTorchDataset(
        dataset_path, cap_val, img_processor, n_samples=n_val, device=device, subcategory=subcategory,
    )
    return data_val


def init_model_and_data(
    component_config: modelComponents,
    dataset_path: str,
    device: torch.device,
    n_val: int = -1,
    checkpoint: str = None,
    init_data: bool = True,
    subcategory: bool = False,
    max_text_length: int = 64,
    min_text_length: int = 48,
):
    # load models and their configs from pretrained checkpoints
    if checkpoint is None:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            component_config.encoder_checkpoint, component_config.decoder_checkpoint
        )
    else:
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    # set decoder config to causal lm
    model.config.decoder_is_decoder = True
    model.config.decoder_add_cross_attention = True
    # set img_processor & tokenizer
    img_processor = component_config.img_processor.from_pretrained(component_config.encoder_checkpoint)
    tokenizer = component_config.tokenizer.from_pretrained(component_config.decoder_checkpoint)
    # decoder-specific config
    if component_config.decoder == BertModel:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.decoder_bos_token_id = tokenizer.cls_token_id
        model.config.decoder_eos_token_id = tokenizer.sep_token_id
    else:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.decoder_bos_token_id = tokenizer.bos_token_id
        model.config.decoder_eos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.decoder.config.pad_token_id = tokenizer.pad_token_id
    model.config.encoder.pad_token_id = tokenizer.pad_token_id
    # generation arguments
    model.config.decoder.repetition_penalty = 10.0
    model.config.decoder.no_repeat_ngram_size = None
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id
    model.config.decoder.do_sample = False
    model.config.decoder.max_length = max_text_length
    model.config.min_length = min_text_length

    # load and prepare data
    if init_data:
        data_val = load_data(dataset_path, tokenizer, img_processor, n_val, device, subcategory)
        return model, tokenizer, data_val
    else:
        return model, tokenizer


# ## Generate captions


def generate_caption(
    model,
    pixel_values,
    num_beams: int = 5,
    do_sample: bool = False,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 10.0,
    max_length: int = 64,
    temperature: int = 1.0,
):
    return model.generate(
        pixel_values,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        # no_repeat_ngram_size=3,
        decoder_start_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_length=max_length,
        # bad_words_ids=[tokenizer.eos_token_id]
    )


# ## Evaluation metrics
bleu_metric = load_metric("sacrebleu")
meteor_metric = load_metric("meteor")
rouge_metric = load_metric("rouge")
bertscore_metric = load_metric("bertscore")


def compute_metrics(eval_preds, decode: bool = True):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    if decode:
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # meteor
    meteor = meteor_metric.compute(predictions=preds, references=labels)
    # rougeL
    rouge = rouge_metric.compute(predictions=preds, references=labels)
    # bertscore
    bertscore = bertscore_metric.compute(predictions=preds, references=labels, lang="en")
    # split into list of tokens and remove spaces
    preds = [pred.split(" ") for pred in preds]
    labels = [[label.split(" ")] for label in labels]
    # bleu
    bleu = bleu_metric.compute(predictions=preds, references=labels)
    result = {
        "bleu": bleu["score"],
        "meteor": meteor["meteor"] * 100,
        "rougeL": rouge["rougeL"][1][2] * 100,
        "bertscore": bertscore,
    }
    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}
    return result


LOSS_T = Literal["triplet", "entropy"]
model, tokenizer, data_val = init_model_and_data(
    vit_gpt2,
    validation_dataset_path,
    device=device,
    checkpoint=checkpoint,
    n_val=-1,
    subcategory=False,
    max_text_length=decoder_max_text_length,
    min_text_length=decoder_min_text_length,
)
model = model.to(device)


def generate_predictions(model, model_label: str, do_sample: bool = False):
    predicted = list()
    rem = len(data_val) % batch_size

    # TODO: iterate dataset with dataloader
    for i in tqdm(range(0, len(data_val) - rem, batch_size)):
        batch = torch.zeros((batch_size, 3, 224, 224)).to(device)
        l = 0
        for j in range(i, i + batch_size):
            batch[l] = data_val[j]["pixel_values"]
            l += 1
        # batch = torch.tensor(batch, device=device, dtype=torch.float)
        predicted.append(generate_caption(model, batch, do_sample=do_sample).cpu())
    if rem != 0:
        for i in range(len(data_val) - rem, len(data_val)):
            predicted.append(
                generate_caption(
                    model, torch.unsqueeze(data_val[i]["pixel_values"], 0).to(device), do_sample=do_sample
                ).cpu()
            )
    predicted = np.asarray(predicted)
    # save to disk
    with open(os.path.join(drive_path, f"predictions/pred-{model_label}.npy"), "wb") as file:
        np.save(file, predicted)


def get_metrics(model_label: str, load_metrics: bool = False):
    with open(drive_path + f"predictions/pred-{model_label}.npy", "rb") as file:
        predicted = np.load(file, allow_pickle=True)
    predicted = [item for sublist in predicted for item in sublist]
    # predicted = pd.Series(map(lambda cap: tokenizer.batch_decode(cap, skip_special_tokens=True), predicted))
    # REAL CAPTIONS
    cap_val = list()
    for p in tqdm(FashionGenDataset(validation_dataset_path).raw_h5()["input_description"], position=0, leave=True):
        cap_val.append(p[0].decode(DEFAULT_STRINGS_ENCODING))  # DEFUALT_STRINGS_ENCODING = DEFAULT_STRINGS_ENCODING)
    cap_val = cap_val[0 : len(predicted)]
    # CATEGORIES
    cat_val = list()
    for p in tqdm(FashionGenDataset(validation_dataset_path).raw_h5()["input_category"], position=0, leave=True):
        cat_val.append(p[0].decode(DEFAULT_STRINGS_ENCODING))  # DEFUALT_STRINGS_ENCODING = DEFAULT_STRINGS_ENCODING)
    cat_val = pd.Series(cat_val[0 : len(predicted)])
    # DATAFRAME
    data = pd.DataFrame({"caption": predicted, "category": cat_val})
    # METRICS
    if load_metrics:
        with open(os.path.join(drive_path, f"predictions/metrics-{model_label}.npy"), "wb") as file:
            scores = np.load(file, allow_pickle=True)
    else:
        scores = {}
        scores["bleu"] = []
        scores["rougeL"] = []
        scores["meteor"] = []
        scores["bertscore"] = []
        for i in tqdm(range(0, len(data.caption.values))):
            # print(data.caption.values[i])
            score = compute_metrics(
                [
                    torch.unsqueeze(data.caption.values[i], 0).cpu(),
                    torch.unsqueeze(torch.tensor(tokenizer.encode(cap_val[i])), 0).cpu(),
                ],
                decode=True,
            )
            # score = compute_metrics([[data.caption.values[i]], [cap_val[i]]], decode=False)
            scores["bleu"].append(score["bleu"])
            scores["meteor"].append(score["meteor"])
            scores["rougeL"].append(score["rougeL"])
            scores["bertscore"].append(score["bertscore"])
            # avg_score = sum(score.values()) / len(score)
            # scores['avg'].append(avg_score)
        with open(os.path.join(drive_path, f"predictions/metrics-{model_label}.npy"), "wb") as file:
            np.save(file, scores)
    # RETURN FINAL DATASET
    # data['score'] = scores
    # data['real'] = cap_val
    # return data


def print_metrics(model_label: str):
    # TRIPLET
    with open(os.path.join(drive_path, f"predictions/metrics-{model_label}.npy"), "wb") as file:
        metrics_triplet = np.load(file, allow_pickle=True).item()
    metrics_triplet = pd.Series(metrics_triplet)
    metrics_triplet.bleu = [val if val != 0 else np.nan for val in metrics_triplet.bleu]
    metrics_triplet.meteor = [val if val != 0 else np.nan for val in metrics_triplet.meteor]
    metrics_triplet.rougeL = [val if val != 0 else np.nan for val in metrics_triplet.rougeL]
    metrics_triplet.bertscore = [val if val != 0 else np.nan for val in metrics_triplet.bertscore]
    with open(os.path.join(drive_path, f"predictions/pred-{model_label}.npy"), mode="rb") as file:
        predictions_triplet = np.load(file, allow_pickle=True)
    predictions_triplet = pd.Series(predictions_triplet)

    # TODO: need to remove this: we only print metrics of the model we want to test. For now load the same things as above
    # ENTROPY
    with open(os.path.join(drive_path, f"predictions/metrics-{model_label}.npy"), "wb") as file:
        metrics_entropy = np.load(file, allow_pickle=True).item()
    metrics_entropy = pd.Series(metrics_entropy)
    metrics_entropy.bleu = [val if val != 0 else np.nan for val in metrics_entropy.bleu]
    metrics_entropy.meteor = [val if val != 0 else np.nan for val in metrics_entropy.meteor]
    metrics_entropy.rougeL = [val if val != 0 else np.nan for val in metrics_entropy.rougeL]
    metrics_entropy.bertscore = [val if val != 0 else np.nan for val in metrics_entropy.bertscore]
    with open(os.path.join(drive_path, f"predictions/pred-{model_label}.npy"), mode="rb") as file:
        predictions_entropy = np.load(file, allow_pickle=True)
    predictions_entropy = pd.Series(predictions_entropy)
    metrics_triplet.bertf1 = []
    metrics_triplet.bertprecision = []
    metrics_triplet.bertrecall = []
    metrics_entropy.bertf1 = []
    metrics_entropy.bertprecision = []
    metrics_entropy.bertrecall = []
    for i in range(0, len(metrics_triplet.bertscore)):
        if metrics_triplet.bertscore[i]["f1"][0] == 0:
            metrics_triplet.bertf1.append(np.nan)
            metrics_triplet.bertprecision.append(np.nan)
            metrics_triplet.bertrecall.append(np.nan)
        else:
            metrics_triplet.bertf1.append(metrics_triplet.bertscore[i]["f1"][0])
            metrics_triplet.bertprecision.append(metrics_triplet.bertscore[i]["precision"][0])
            metrics_triplet.bertrecall.append(metrics_triplet.bertscore[i]["recall"][0])
    for i in range(0, len(metrics_entropy.bertscore)):
        if metrics_entropy.bertscore[i]["f1"][0] == 0:
            metrics_entropy.bertf1.append(np.nan)
            metrics_entropy.bertprecision.append(np.nan)
            metrics_entropy.bertrecall.append(np.nan)
        else:
            metrics_entropy.bertf1.append(metrics_entropy.bertscore[i]["f1"][0])
            metrics_entropy.bertprecision.append(metrics_entropy.bertscore[i]["precision"][0])
            metrics_entropy.bertrecall.append(metrics_entropy.bertscore[i]["recall"][0])
    print(
        pd.DataFrame(
            {
                "Bleu": [round(np.nanmean(metrics_entropy.bleu), 1), round(np.nanmean(metrics_triplet.bleu), 1)],
                "Meteor": [round(np.nanmean(metrics_entropy.meteor), 1), round(np.nanmean(metrics_triplet.meteor), 1)],
                "RougeL": [round(np.nanmean(metrics_entropy.rougeL), 1), round(np.nanmean(metrics_triplet.rougeL), 1)],
                "BertScore(f1)": [
                    round(np.nanmean(metrics_entropy.bertf1) * 100, 3),
                    round(np.nanmean(metrics_triplet.bertf1) * 100, 3),
                ],
                "BertScore(precision)": [
                    round(np.nanmean(metrics_entropy.bertprecision) * 100, 3),
                    round(np.nanmean(metrics_triplet.bertprecision) * 100, 3),
                ],
                "BertScore(recall)": [
                    round(np.nanmean(metrics_entropy.bertrecall) * 100, 3),
                    round(np.nanmean(metrics_triplet.bertrecall) * 100, 3),
                ],
            },
            index=["Entropy", "Triplet"],
        )
    )


if regenerate_predictions:
    generate_predictions(model, model_label=checkpoint_name)
if regenerate_metrics:
    get_metrics(load_metrics=False, model_label=checkpoint_name)
else:
    get_metrics(load_metrics=True, model_label=checkpoint_name)
if print_metrics:
    print_metrics()

