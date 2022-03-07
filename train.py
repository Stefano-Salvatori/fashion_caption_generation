import h5py
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_metric
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2Model,
    BertModel,
    BertTokenizerFast,
    GPT2TokenizerFast,
    ViTModel,
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
)
from modules.custom_trainer import CustomLossTrainer
from modules.dataset import FashionGenTorchDataset
from modules.fashiongen_utils import FashionGenDataset, DEFAULT_STRINGS_ENCODING
from torch.utils.tensorboard import SummaryWriter

from modules.train_utils import generate_caption

# MAIN PATHS
drive_path = "./"  #'drive/MyDrive/Tesi/'
data_path = "/home/salvatori/datasets/FashionGen/"
checkpoints_path = "./checkpoints/"  # drive_path + 'checkpoints/'
step = 0

# TRAIN CONFIG
loss_type = "triplet"  # triplet
step = 0
batch_size = 4
max_caption_len = 64
checkpoint = None  # checkpoints_path + loss_type + "-checkpoint-" + str(step)

generation_num_beams = 3
save_total_limit = 3
learning_rate = 4e-6
num_train_epochs = 5
warmup_steps = 500
weight_decay = 0.01
predict_with_generate = True

generation_do_sample = False
generation_top_p = 1.0
generation_top_k = 50
generation_repetition_penalty = 10.0
generation_temperature = 1.0

triplet_margin = 0.1
pretrained_text_embedder = "bert-base-uncased"

fashiongen_train_file = "fashiongen_train.h5"
fashiongen_validation_file = "fashiongen_validation.h5"

print(f"Available devices= {torch.cuda.device_count()}")

# ### Tensorboard Monitoring
tensorboard_path = drive_path + "tensorboard/"
log_path = (
    tensorboard_path + "entropy_subcat"
    if loss_type == "entropy"
    else tensorboard_path + "swap_from_scratch_subcat_altnorm_lowmargin_high_lr_test"
)

writer = SummaryWriter(log_dir=log_path)


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
vit_bert = modelComponents(
    encoder=ViTModel,
    encoder_checkpoint="google/vit-base-patch16-224-in21k",
    decoder=BertModel,
    decoder_checkpoint="bert-base-uncased",
    img_processor=ViTFeatureExtractor,
    tokenizer=BertTokenizerFast,
)

vit_gpt2 = modelComponents(
    encoder=ViTModel,
    encoder_checkpoint="google/vit-base-patch16-224-in21k",
    decoder=GPT2Model,
    decoder_checkpoint="gpt2",
    img_processor=ViTFeatureExtractor,
    tokenizer=GPT2TokenizerFast,
)

# ## Device & Batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Model and Data setup
# ## Model and Dataset init


def load_data(tokenizer, img_processor, n_train, n_val, subcategory: bool = False):
    cap_train = list()
    for p in tqdm(FashionGenDataset(data_path + fashiongen_train_file).raw_h5()["input_description"]):
        cap_train.append(p[0].decode(DEFAULT_STRINGS_ENCODING))
    cap_val = list()
    for p in tqdm(FashionGenDataset(data_path + fashiongen_validation_file).raw_h5()["input_description"]):
        cap_val.append(p[0].decode(DEFAULT_STRINGS_ENCODING))
    # append all captions in a single shallow list, tokenize everything
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    cap = list(
        map(
            lambda caption: tokenizer.encode(
                caption.replace("<br>", " "),
                return_tensors="pt",
                max_length=max_caption_len,
                padding="max_length",
                truncation=True,
            )[0],
            cap_train + cap_val,
        )
    )
    # split into train and validation again
    cap_train = cap[0 : len(cap_train)]
    cap_val = cap[len(cap_train) :]
    # create datasets
    data_train = FashionGenTorchDataset(
        data_path + fashiongen_train_file,
        cap_train,
        img_processor,
        n_samples=n_train,
        device=device,
        subcategory=subcategory,
    )
    data_val = FashionGenTorchDataset(
        data_path + fashiongen_validation_file,
        cap_val,
        img_processor,
        n_samples=n_val,
        device=device,
        subcategory=subcategory,
    )
    return data_train, data_val


def init_model_and_data(
    component_config: modelComponents,
    n_train: int = -1,
    n_val: int = -1,
    checkpoint: str = None,
    init_data: bool = True,
    subcategory: bool = False,
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
    model.config.decoder.max_length = max_caption_len
    # load and prepare data
    if init_data:
        data_train, data_val = load_data(tokenizer, img_processor, n_train, n_val, subcategory)
        return model, tokenizer, data_train, data_val
    else:
        return model, tokenizer


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
    vit_gpt2, checkpoint=checkpoint, n_train=-1, n_val=-1, subcategory=True
)
if loss_type == "triplet":
    bert = AutoModel.from_pretrained(pretrained_text_embedder)
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_text_embedder)
    bert = bert.to(device)
else:
    bert, bert_tokenizer = None, None
model = model.to(device)


training_args = Seq2SeqTrainingArguments(
    dataloader_pin_memory=not device.type == "cuda",
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,  # batch size for evaluation
    output_dir=checkpoints_path,  # output directory
    overwrite_output_dir=False,
    load_best_model_at_end=False,
    predict_with_generate=predict_with_generate,
    generation_num_beams=generation_num_beams,
    eval_accumulation_steps=4000,  # send logits and labels to cpu for evaluation step by step, rather than all together
    evaluation_strategy="steps",
    save_strategy="epoch",
    save_total_limit=save_total_limit,  # Only last [save_total_limit] models are saved. Older ones are deleted.
    # save_steps = 1000,
    eval_steps=99999999,  # 16281,    # Evaluation and Save happens every [eval_steps] steps
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,  # total number of training epochs
    warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
    weight_decay=weight_decay,  # strength of weight decay
)


trainer = CustomLossTrainer(
    tokenizer=tokenizer,
    writer=writer,
    step=step,
    triplet_margin=triplet_margin,
    triplet_text_embedder=bert,
    triplet_text_tokenizer=bert_tokenizer,
    max_text_embedding_length=max_caption_len,
    loss_type=loss_type,
    compute_metrics=compute_metrics,
    generation_function=generate_caption,
    data_collator=None,
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=data_train,  # training dataset
    eval_dataset=data_val,  # evaluation dataset
)

trainer.train(checkpoint)
# trainer.train()
writer.flush()
