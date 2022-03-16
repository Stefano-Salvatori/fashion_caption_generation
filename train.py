import random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_metric
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
    SchedulerType,
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
)
from modules.custom_trainer import CustomTrainer
from modules.dataset import FashionGenTorchDataset, NegativeSampleType
from modules.fashiongen_utils import FashionGenDataset, DEFAULT_STRINGS_ENCODING
from dataclasses import fields
from modules.train_utils import GenerationConfig, ModelComponents
from transformers.utils import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.get_logger(__name__)


# MAIN PATHS
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)

data_path = "/home/salvatori/datasets/FashionGen/"
fashiongen_train_file = "fashiongen_train.h5"
fashiongen_validation_file = "fashiongen_validation.h5"
checkpoints_path = "./checkpoints/"  # drive_path + 'checkpoints/'

# TRAIN CONFIG
loss_type = "entropy"
step = 0
train_batch_size = 12
eval_batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4

checkpoint = None  # checkpoints_path + loss_type + "-checkpoint-" + str(step)

save_total_limit = 3
logging_steps = 25
eval_steps = 10000
lr_scheduler_type = SchedulerType.COSINE
learning_rate = 2e-5
num_train_epochs = 5
warmup_steps = 500
weight_decay = 0.01

predict_with_generate = True

generation_config = GenerationConfig(
    max_length=128,
    min_length=0,
    do_sample=False,
    num_beams=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    diversity_penalty=0.0,
    repetition_penalty=10.0,
    length_penalty=1.0,
    no_repeat_ngram_size=0,
    bad_words_ids=None,
)

triplet_margin = 0.1
pretrained_text_embedder = "bert-base-uncased"
negative_sample_type = NegativeSampleType.RANDOM  # NegativeSampleType.SAME_SUBCATEGORY

logger.info(f"Available devices= {torch.cuda.device_count()}")

# ### Tensorboard Monitoring
experiment_name = "entropy"
log_path = os.path.join("tensorboard", experiment_name)
# log_path = os.path.join(tensorboard_path, experiment_name)
# writer = SummaryWriter(log_dir=log_path)


# component configurations
vit_bert = ModelComponents(
    encoder_checkpoint="google/vit-base-patch16-224-in21k",
    decoder_checkpoint="bert-base-uncased",
    img_processor=ViTFeatureExtractor,
    generation_config=generation_config,
)

vit_gpt2 = ModelComponents(
    encoder_checkpoint="google/vit-base-patch16-224-in21k",
    decoder_checkpoint="gpt2",
    img_processor=ViTFeatureExtractor,
    generation_config=generation_config,
)


# # Model and Data setup
# ## Model and Dataset init
def load_data(
    dataset_train_path: str,
    dataset_validation_path: str,
    tokenizer: PreTrainedTokenizer,
    img_processor,
    n_train,
    n_val,
    negative_sample_type: NegativeSampleType,
):
    logger.info("Encoding captions")
    cap_train = list()
    for p in tqdm(FashionGenDataset(dataset_train_path).raw_h5()["input_description"]):
        cap_train.append(p[0].decode(DEFAULT_STRINGS_ENCODING).replace("<br>", " "))
    cap_val = list()
    for p in tqdm(FashionGenDataset(dataset_validation_path).raw_h5()["input_description"]):
        cap_val.append(p[0].decode(DEFAULT_STRINGS_ENCODING).replace("<br>", " "))

    # tokenizer.padding_side = "left"  # TODO: why?
    if not tokenizer.pad_token:
        # we can use the  EOS token as PAD token if the tokenizer doesn't have one
        # (https://huggingface.co/docs/transformers/master/model_doc/vision-encoder-decoder#:~:text=model.config.pad_token_id%20%3D%20model.config.eos_token_id)
        tokenizer.pad_token = tokenizer.eos_token

    cap_train = tokenizer.batch_encode_plus(cap_train, return_tensors="pt", padding=True)
    cap_val = tokenizer.batch_encode_plus(cap_val, return_tensors="pt", padding=True)

    logger.info("Creating torch train/validation datasets")
    data_train = FashionGenTorchDataset(
        dataset_train_path,
        cap_train,
        img_processor,
        n_samples=n_train,
        device=device,
        negative_sample_type=negative_sample_type,
    )
    data_val = FashionGenTorchDataset(
        dataset_validation_path,
        cap_val,
        img_processor,
        n_samples=n_val,
        device=device,
        negative_sample_type=negative_sample_type,
    )
    return data_train, data_val


def init_model_and_data(
    component_config: ModelComponents,
    dataset_train_path: str,
    dataset_validation_path: str,
    n_train: int = -1,
    n_val: int = -1,
    checkpoint: str = None,
    init_data: bool = True,
    negative_sample_type: NegativeSampleType = NegativeSampleType.RANDOM,
):
    logger.info("Initializing model")
    # load models and their configs from pretrained checkpoints
    if checkpoint is None:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            component_config.encoder_checkpoint, component_config.decoder_checkpoint
        )
    else:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    # Should be automatically set by previous methods
    # set decoder config to causal lm
    # model.config.decoder_is_decoder = True
    # model.config.decoder_add_cross_attention = True

    model.config.decoder_start_token_id = component_config.tokenizer.bos_token_id
    model.config.decoder_bos_token_id = component_config.tokenizer.bos_token_id
    model.config.decoder_eos_token_id = component_config.tokenizer.eos_token_id

    # we can use the  EOS token as PAD token if the tokenizer doesn't have one
    # (https://huggingface.co/docs/transformers/master/model_doc/vision-encoder-decoder#:~:text=model.config.pad_token_id%20%3D%20model.config.eos_token_id)
    model.config.pad_token_id = (
        component_config.tokenizer.eos_token_id
        if not component_config.tokenizer.pad_token_id
        else component_config.tokenizer.pad_token_id
    )

    model.config.decoder_start_token_id = component_config.tokenizer.bos_token_id
    model.config.decoder.eos_token_id = component_config.tokenizer.eos_token_id
    # component_config.tokenizer.pad_token = component_config.tokenizer.eos_token

    # model.decoder.config.pad_token_id = component_config.tokenizer.pad_token_id
    # model.config.encoder.pad_token_id = component_config.tokenizer.pad_token_id

    # set generation arguments
    for field in fields(component_config.generation_config):
        setattr(model.config.decoder, field.name, getattr(component_config.generation_config, field.name))

    # load and prepare data
    if init_data:
        data_train, data_val = load_data(
            dataset_train_path=dataset_train_path,
            dataset_validation_path=dataset_validation_path,
            tokenizer=component_config.tokenizer,
            img_processor=component_config.img_processor,
            n_train=n_train,
            n_val=n_val,
            negative_sample_type=negative_sample_type,
        )
        return model, component_config.tokenizer, data_train, data_val
    else:
        return model, component_config.tokenizer


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


# ## Triplet-Loss Test
# ### Model and Data setup
model, tokenizer, data_train, data_val = init_model_and_data(
    component_config=vit_gpt2,
    dataset_train_path=os.path.join(data_path, fashiongen_train_file),
    dataset_validation_path=os.path.join(data_path, fashiongen_validation_file),
    checkpoint=checkpoint,
    n_train=-1,
    n_val=-1,  # -1
    negative_sample_type=negative_sample_type,
)
if loss_type == "triplet":
    # TODO: replace manual bert encoding with huggingface pipeline ("feature-extraction")
    bert = AutoModel.from_pretrained(pretrained_text_embedder)
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_text_embedder)
    bert = bert.to(device)
else:
    bert, bert_tokenizer = None, None
model = model.to(device)

training_args = Seq2SeqTrainingArguments(
    dataloader_pin_memory=not device.type == "cuda",
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
    output_dir=checkpoints_path,  # output directory,
    logging_dir=log_path,
    logging_steps=logging_steps,
    run_name=experiment_name,
    load_best_model_at_end=False,
    predict_with_generate=predict_with_generate,
    generation_num_beams=generation_config.num_beams,
    # eval_accumulation_steps=2,  # send logits and labels to cpu for evaluation step by step, rather than all together
    evaluation_strategy="steps",
    save_strategy="epoch",
    save_total_limit=save_total_limit,  # Only last [save_total_limit] models are saved. Older ones are deleted.
    # save_steps = 1000,
    eval_steps=eval_steps,  # 16281,    # Evaluation and Save happens every [eval_steps] steps
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    num_train_epochs=num_train_epochs,  # total number of training epochs
    warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
    weight_decay=weight_decay,  # strength of weight decay
    seed=random_seed,
    dataloader_num_workers=num_workers,
)


trainer = CustomTrainer(
    tokenizer=tokenizer,
    generation_config=generation_config,
    triplet_margin=triplet_margin,
    triplet_text_embedder=bert,
    triplet_text_tokenizer=bert_tokenizer,
    max_text_embedding_length=generation_config.max_length,
    loss_type=loss_type,
    compute_metrics=compute_metrics,
    # generation_function=generate_caption,
    # data_collator=None,
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=data_train,  # training dataset
    eval_dataset=data_val,  # evaluation dataset
)

trainer.train(checkpoint)
# trainer.train()
# writer.flush()

