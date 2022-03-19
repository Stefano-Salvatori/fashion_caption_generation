import random
import torch
from datasets import load_metric
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    SchedulerType,
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
)
from modules.callbacks import GPUStatsMonitor
from modules.custom_trainer import CustomTrainer
from modules.data.dataset import FashionGenTorchDataset, NegativeSampleType
from dataclasses import fields
from modules.metrics.metrics_utils import compute_metrics
from modules.train_utils import GenerationConfig, ModelComponents
from transformers.utils import logging
from functools import partial
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.get_logger(__name__)

# TODO: Handle configurations with other libraries (e.g., abseil, sacred) so that some parameters can be set from
# command line
### CONFIGURATIONS ###

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)

data_path = "/home/salvatori/datasets/FashionGen/"
fashiongen_train_file = "fashiongen_train.h5"
fashiongen_validation_file = "fashiongen_validation.h5"
train_dataset_size = -1  # set -1 to use whole dataset
validation_dataset_size = -1  # set -1 to use whole dataset

# TRAIN CONFIG
loss_type = "entropy"
step = 0
train_batch_size = 32
eval_batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 3

checkpoint = None  # checkpoints_path + loss_type + "-checkpoint-" + str(step)

save_total_limit = 5
patience = 4
save_steps = 500
logging_steps = 25
lr_scheduler_type = SchedulerType.COSINE
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01
num_train_epochs = 10

predict_with_generate = True

max_captions_length = 128
generation_config = GenerationConfig(
    max_length=max_captions_length,
    min_length=0,
    do_sample=False,
    num_beams=1,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    diversity_penalty=0.0,
    repetition_penalty=5.0,
    length_penalty=1.0,
    no_repeat_ngram_size=0,
    bad_words_ids=None,
)

triplet_margin = 0.1
pretrained_text_embedder = "bert-base-uncased"
negative_sample_type = NegativeSampleType.RANDOM  # NegativeSampleType.SAME_SUBCATEGORY

# Tensorboard Monitoring
experiment_name = "entropy_10epoch"
experiment_name = f"{experiment_name}_{time.strftime('%d%m%H%M')}"
log_path = os.path.join("tensorboard", experiment_name)
checkpoints_path = os.path.join("checkpoints", experiment_name)  # drive_path + 'checkpoints/'


# Evaluation metrics
validation_metrics = ["sacrebleu", "meteor", "rouge"]  # , "modules/metrics/eng_bertscore.py"]
validation_metrics = [load_metric(v) for v in validation_metrics]

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


def init_model_and_data(
    component_config: ModelComponents,
    dataset_train_path: str,
    dataset_validation_path: str,
    n_train: int = -1,
    n_val: int = -1,
    checkpoint: str = None,
    max_captions_length: int = 256,
    negative_sample_type: NegativeSampleType = NegativeSampleType.RANDOM,
):
    # load models and their configs from pretrained checkpoints
    if checkpoint is None:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            component_config.encoder_checkpoint, component_config.decoder_checkpoint
        )
    else:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    model.config.decoder_is_decoder = True
    model.config.decoder_add_cross_attention = True
    model.config.decoder_start_token_id = (
        component_config.tokenizer.bos_token_id
        if component_config.tokenizer.bos_token_id is not None
        else component_config.tokenizer.cls_token_id
    )

    # we can use the  EOS token as PAD token if the tokenizer doesn't have one
    # (https://huggingface.co/docs/transformers/master/model_doc/vision-encoder-decoder#:~:text=model.config.pad_token_id%20%3D%20model.config.eos_token_id)
    model.config.pad_token_id = (
        component_config.tokenizer.pad_token_id
        if component_config.tokenizer.pad_token_id is not None
        else component_config.tokenizer.eos_token_id
    )
    model.config.decoder_bos_token_id = model.config.decoder_start_token_id
    model.config.decoder_eos_token_id = component_config.tokenizer.eos_token_id

    # set generation arguments
    for field in fields(component_config.generation_config):
        setattr(model.config.decoder, field.name, getattr(component_config.generation_config, field.name))

    # create torch dataset
    data_train = FashionGenTorchDataset(
        file_name=dataset_train_path,
        text_tokenizer=component_config.tokenizer,
        img_processor=component_config.img_processor,
        n_samples=n_train,
        max_text_length=max_captions_length,
        negative_sample_type=negative_sample_type,
    )
    data_val = FashionGenTorchDataset(
        file_name=dataset_validation_path,
        text_tokenizer=component_config.tokenizer,
        img_processor=component_config.img_processor,
        n_samples=n_val,
        max_text_length=max_captions_length,
        negative_sample_type=negative_sample_type,
    )
    return model, component_config.tokenizer, data_train, data_val

# ### Model and Data setup
model, tokenizer, data_train, data_val = init_model_and_data(
    component_config=vit_gpt2,
    dataset_train_path=os.path.join(data_path, fashiongen_train_file),
    dataset_validation_path=os.path.join(data_path, fashiongen_validation_file),
    checkpoint=checkpoint,
    n_train=train_dataset_size,
    n_val=validation_dataset_size,
    max_captions_length=max_captions_length,
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
    evaluation_strategy="epoch",
    save_strategy="steps",
    save_total_limit=save_total_limit,  # Only last [save_total_limit] models are saved. Older ones are deleted.
    save_steps=save_steps,
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
    compute_metrics=partial(compute_metrics, validation_metrics=validation_metrics, tokenizer=tokenizer),
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=data_train,  # training dataset
    eval_dataset=data_val,  # evaluation dataset
)

callbacks = [EarlyStoppingCallback(early_stopping_patience=patience), GPUStatsMonitor()]

trainer.train(resume_from_checkpoint=checkpoint)
# trainer.train()
# writer.flush()

