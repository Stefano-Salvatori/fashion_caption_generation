import random
import torch
from datasets import load_metric
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    SchedulerType,
    ViTFeatureExtractor,
    Seq2SeqTrainingArguments,
)
from modules.callbacks import GPUStatsMonitor
from modules.custom_trainer import CustomTrainer
from modules.data.dataset import FashionGenTorchDataset, NegativeSampleType
from modules.metrics.metrics_utils import compute_metrics
from modules.train_utils import GenerationConfig, LossType, ModelComponents, init_model
from transformers.utils import logging
from transformers.trainer_utils import PredictionOutput
from functools import partial
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.get_logger(__name__)

# TODO: Handle configurations with other libraries (e.g., abseil, sacred) so that some parameters can be set from
# command line
### CONFIGURATIONS ###

type = "TRAIN"  # TRAIN
loss_type = LossType.ENTROPY_TRIPLET

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)

data_path = "/home/salvatori/datasets/FashionGen/"
fashiongen_train_file = "fashiongen_train.h5"
fashiongen_validation_file = "fashiongen_validation.h5"
train_dataset_size = -1  # set -1 to use whole dataset
validation_dataset_size = -1  # set -1 to use whole dataset

# TRAIN CONFIG
step = 0
train_batch_size = 16
eval_batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4


save_total_limit = 4
patience = 3
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
    repetition_penalty=1.0,
    length_penalty=1.0,
    no_repeat_ngram_size=0,
    bad_words_ids=None,
    early_stopping=True,
)

triplet_margin = 1.0
pretrained_text_embedder = "bert-base-uncased"
negative_sample_type = NegativeSampleType.SAME_SUBCATEGORY  # NegativeSampleType.SAME_SUBCATEGORY

# Tensorboard Monitoring
experiment_name = "entropy_triplet_10epoch"
log_path = os.path.join("tensorboard", experiment_name)
checkpoints_path = os.path.join("checkpoints", experiment_name)  # drive_path + 'checkpoints/'

# checkpoint = "./checkpoints/entropy_10epoch_fixeos_2/checkpoint-40705"  
checkpoint = None

# Evaluation metrics
validation_metrics = ["sacrebleu", "meteor", "rouge"] #, "modules/metrics/eng_bertscore.py"]
validation_metrics = [load_metric(v) for v in validation_metrics]

# component configurations
# vit_bert = ModelComponents(
#     encoder_checkpoint="google/vit-base-patch16-224-in21k",
#     decoder_checkpoint="bert-base-uncased",
#     img_processor=ViTFeatureExtractor,
#     generation_config=generation_config,
# )

encoder_decoder_components = ModelComponents(
    encoder_checkpoint="google/vit-base-patch16-224-in21k",
    decoder_checkpoint="gpt2",
    img_processor=ViTFeatureExtractor,
    generation_config=generation_config,
)

model = init_model(encoder_decoder_components, checkpoint)
model = model.to(device)

# create torch train and validation dataset
data_train = FashionGenTorchDataset(
    file_name=os.path.join(data_path, fashiongen_train_file),
    text_tokenizer=encoder_decoder_components.tokenizer,
    img_processor=encoder_decoder_components.img_processor,
    n_samples=train_dataset_size,
    max_text_length=max_captions_length,
    sample_negative=loss_type == LossType.ENTROPY_TRIPLET,
    negative_sample_type=negative_sample_type,
)
data_val = FashionGenTorchDataset(
    file_name=os.path.join(data_path, fashiongen_validation_file),
    text_tokenizer=encoder_decoder_components.tokenizer,
    img_processor=encoder_decoder_components.img_processor,
    n_samples=validation_dataset_size,
    max_text_length=max_captions_length,
    sample_negative=loss_type == LossType.ENTROPY_TRIPLET,
    negative_sample_type=negative_sample_type,
)

if loss_type == LossType.ENTROPY_TRIPLET:
    # TODO: replace manual bert encoding with huggingface pipeline ("feature-extraction")
    bert = AutoModel.from_pretrained(pretrained_text_embedder)
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_text_embedder)
    bert = bert.to(device)
else:
    bert, bert_tokenizer = None, None

training_args = Seq2SeqTrainingArguments(
    dataloader_pin_memory=not device.type == "cuda",
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
    output_dir=checkpoints_path,  # output directory,
    logging_dir=log_path,
    logging_steps=logging_steps,
    run_name=experiment_name,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=predict_with_generate,
    generation_num_beams=generation_config.num_beams,
    evaluation_strategy="epoch",
    save_strategy="epoch",
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

callbacks = [EarlyStoppingCallback(early_stopping_patience=patience), GPUStatsMonitor()]

trainer = CustomTrainer(
    tokenizer=encoder_decoder_components.tokenizer,
    generation_config=generation_config,
    triplet_margin=triplet_margin,
    triplet_text_embedder=pretrained_text_embedder,
    loss_type=loss_type,
    compute_metrics=partial(
        compute_metrics, validation_metrics=validation_metrics, tokenizer=encoder_decoder_components.tokenizer
    ),
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=data_train,  # training dataset
    eval_dataset=data_val,  # evaluation dataset
    callbacks=callbacks,
)


if type == "TRAIN":
    trainer.train(resume_from_checkpoint=checkpoint)
elif type == "TEST":
    predictions: PredictionOutput = trainer.predict(test_dataset=data_val)
    trainer.save_metrics("test", predictions.metrics)
    print(predictions.metrics)

# trainer.train()
# writer.flush()

