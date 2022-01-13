import h5py
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_metric
from transformers import (GPT2Model, BertModel, BertTokenizerFast, GPT2TokenizerFast, ViTModel, ViTFeatureExtractor,
                          VisionEncoderDecoderModel, Seq2SeqTrainingArguments)
from modules.custom_trainer import CustomTrainer
from modules.fashiongen_utils import FashionGenDataset
from torch.utils.tensorboard import SummaryWriter

# MAIN PATHS
drive_path = './' #'drive/MyDrive/Tesi/'
data_path = '../../../../mnt/ssd/salvatori/datasets/FashionGen/'
checkpoints_path = './checkpoints/' #drive_path + 'checkpoints/'

# class to hold components of Encoder-Decoder model
class modelComponents():
  def __init__(self, encoder, encoder_checkpoint, decoder, decoder_checkpoint, img_processor, tokenizer):
    self.encoder_checkpoint = encoder_checkpoint
    self.decoder_checkpoint = decoder_checkpoint
    self.img_processor = img_processor
    self.tokenizer = tokenizer
    self.encoder = encoder
    self.decoder = decoder

# component configurations
vit_bert = modelComponents(encoder=ViTModel, encoder_checkpoint='google/vit-base-patch16-224-in21k',
                           decoder=BertModel, decoder_checkpoint='bert-base-uncased',
                           img_processor=ViTFeatureExtractor, tokenizer=BertTokenizerFast)

vit_gpt2 = modelComponents(encoder=ViTModel, encoder_checkpoint='google/vit-base-patch16-224-in21k',
                           decoder=GPT2Model, decoder_checkpoint='gpt2',
                           img_processor=ViTFeatureExtractor, tokenizer=GPT2TokenizerFast)

# ## Device & Batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16

# # Model and Data setup
# ## Dataset class

fashiongen_train = FashionGenDataset(data_path + "fashiongen_train.h5")

class FashionGenTorchDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, caption_encodings, img_processor, n_samples:int, subcategory:bool=False):
        self.n_samples = n_samples
        self.file_path = file_name
        self.dataset = h5py.File(file_name, mode='r')["input_image"]
        self.caption_encodings = caption_encodings
        self.img_processor = img_processor
        self.subcategory = subcategory
        if(self.n_samples == -1): self.n_samples = len(self.dataset)
        else: assert n_samples <= len(self.dataset), 'n_samples must be <=' + str(len(self.dataset))

    def __getitem__(self, idx):
        return {"pixel_values":self.preprocess_image(self.dataset[idx]),
                "labels":self.caption_encodings[idx],
                "negative":self.get_product_same_category(idx)}

    def set_subcategory(self, subcategory:bool):
        self.subcategory = subcategory

    def __len__(self):
        return self.n_samples

    def preprocess_image(self, image):
        return self.img_processor(image, return_tensors="pt")['pixel_values'][0]

    def get_product_same_category(self, index:int):
        product = fashiongen_train.get_product(index)
        if self.subcategory:
          product.subcategory = product.subcategory.encode("ISO-8859-9")
          similiar = fashiongen_train.get_same_subcategory_of(product)[0].image
        else :
          product.category = product.category.encode("ISO-8859-9")
          similiar = fashiongen_train.get_same_category_of(product)[0].image
        return self.preprocess_image(similiar).to(device)

# ## Model and Dataset init

def load_data(tokenizer, img_processor, n_train, n_val, subcategory:bool=False):
  cap_train = list()
  for p in tqdm(FashionGenDataset(data_path + "fashiongen_train.h5").raw_h5()["input_description"], position=0, leave=True):
      cap_train.append(p[0].decode("ISO-8859-9")) #DEFUALT_STRINGS_ENCODING = "ISO-8859-9")

  cap_val = list()
  for p in tqdm(FashionGenDataset(data_path + "fashiongen_validation.h5").raw_h5()["input_description"], position=0, leave=True):
      cap_val.append(p[0].decode("ISO-8859-9")) #DEFUALT_STRINGS_ENCODING = "ISO-8859-9")

  # append all captions in a single shallow list, tokenize everything
  tokenizer.padding_side = "left"
  tokenizer.pad_token = tokenizer.eos_token
  cap = list(map(lambda caption: tokenizer.encode(caption.replace('<br>', ' '), return_tensors="pt", max_length=64, pad_to_max_length=True, truncation=True)[0], cap_train + cap_val))

  # split into train and validation again
  cap_train = cap[0:len(cap_train)]
  cap_val = cap[len(cap_train):]

  # create datasets
  data_train = FashionGenTorchDataset(data_path + "fashiongen_train.h5", cap_train, img_processor, n_samples=n_train, subcategory=subcategory)
  data_val = FashionGenTorchDataset(data_path + "fashiongen_validation.h5", cap_val, img_processor, n_samples=n_val, subcategory=subcategory)
  
  return data_train, data_val

def init_model_and_data(component_config:modelComponents, n_train:int=-1, n_val:int=-1, checkpoint:str=None, init_data:bool=True, subcategory:bool=False):
  # load models and their configs from pretrained checkpoints
  if(checkpoint is None):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(component_config.encoder_checkpoint, component_config.decoder_checkpoint)
  else:
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)

  # set decoder config to causal lm
  model.config.decoder_is_decoder = True
  model.config.decoder_add_cross_attention = True

  # set img_processor & tokenizer
  img_processor = component_config.img_processor.from_pretrained(component_config.encoder_checkpoint)
  tokenizer = component_config.tokenizer.from_pretrained(component_config.decoder_checkpoint)

  # decoder-specific config
  if(component_config.decoder == BertModel):
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
  model.config.decoder.max_length = 64

  # load and prepare data
  if(init_data):
    data_train, data_val = load_data(tokenizer, img_processor, n_train, n_val, subcategory)
    return model, tokenizer, data_train, data_val
  else:
    return model, tokenizer

# ## Generate captions

def generate_caption(model, pixel_values, num_beams:int=3, do_sample:bool=False, top_p:float=1.0, top_k:int=50, repetition_penalty:float=10.0, max_length:int=64, temperature:int=1.0):
    return model.generate(pixel_values,
                          num_beams=num_beams,
                          repetition_penalty=repetition_penalty,
                          #no_repeat_ngram_size=3,
                          decoder_start_token_id=tokenizer.bos_token_id,
                          pad_token_id=tokenizer.pad_token_id,
                          eos_token_id=tokenizer.eos_token_id,
                          do_sample=do_sample,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p,
                          max_length=max_length,
                          #bad_words_ids=[tokenizer.eos_token_id]
                          )

# ## Evaluation metrics
bleu_metric = load_metric('sacrebleu')
meteor_metric = load_metric('meteor')
rouge_metric = load_metric('rouge')

def compute_metrics(eval_preds, decode:bool=True):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    if(decode):
      preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
      # Replace -100 in the labels as we can't decode them.
      labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
      labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # meteor
    meteor = meteor_metric.compute(predictions=preds, references=labels)

    # rougeL
    rouge = rouge_metric.compute(predictions=preds, references=labels)

    #split into list of tokens and remove spaces
    preds = [pred.split(' ') for pred in preds]
    labels = [[label.split(' ')] for label in labels]

    # bleu
    bleu = bleu_metric.compute(predictions=preds, references=labels)
    
    result = {"bleu": bleu["score"], "meteor": meteor['meteor']*100, "rougeL": rouge['rougeL'][1][2]*100}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# ## Triplet-Loss Test
# ### Model and Data setup
step = 0
checkpoint = None#checkpoints_path + 'checkpoint-' + str(step)
model, tokenizer, data_train, data_val = init_model_and_data(vit_gpt2, checkpoint=checkpoint, n_train=-1, n_val=-1, subcategory=True)
bert = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = model.to(device)
bert = bert.to(device)

# ### Tensorboard Monitoring
tensorboard_path = drive_path+'tensorboard/'
log_path = tensorboard_path+"swap_from_scratch_subcat_altnorm_lowmargin"

writer = SummaryWriter(log_dir=log_path)

# ### Loss and Trainer config

def normalize_0_to_1(x:torch.Tensor):
  # x -= x.min(1, keepdim=True)[0]
  # x /= x.max(1, keepdim=True)[0]
  # writer.add_text('tensor_shape', str(list(x.shape)))
  return torch.nn.functional.normalize(x, dim=1)

def get_bert_embedding(text, normalize:bool=True, decode:bool=True):
  if(decode):
    text = tokenizer.batch_decode(text, skip_special_tokens=True)
  input_ids = bert_tokenizer(text, truncation=True, max_length=64, padding='max_length', return_tensors='pt')['input_ids'].to(device)
  with torch.no_grad():
    embedding = bert(input_ids)['pooler_output']
  embedding.requires_grad
  return normalize_0_to_1(embedding) if normalize else embedding

def get_encoder_embedding(pixel_values, normalize:bool=True):
  embedding = model.encoder(pixel_values.to(device))['pooler_output']
  return normalize_0_to_1(embedding) if normalize else embedding

def triplet_margin_loss(pixel_values, negatives, swap:bool=True):
    negative_embeddings = get_encoder_embedding(negatives)
    positive_embeddings = get_encoder_embedding(pixel_values)
    captions = generate_caption(model, pixel_values)
    caption_embeddings = get_bert_embedding(captions)
    return torch.nn.functional.triplet_margin_loss(anchor=caption_embeddings, positive=positive_embeddings, negative=negative_embeddings, margin=0.1, swap=swap)

# ### CustomLossTrainer
class CustomLossTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = step

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        negative = inputs.pop("negative")
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            entropy_loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            entropy_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        triplet_loss = triplet_margin_loss(inputs["pixel_values"], negative)
        final_loss = entropy_loss + triplet_loss

        writer.add_scalar("Loss/train/triplet", torch.mean(triplet_loss), self.step)
        writer.add_scalar("Loss/train/entropy", torch.mean(entropy_loss), self.step)
        writer.add_scalar("Loss/train/final", torch.mean(final_loss), self.step)
        self.step += 1

        return (final_loss, outputs) if return_outputs else final_loss

training_args = Seq2SeqTrainingArguments(
    dataloader_pin_memory = not device.type == 'cuda',
    per_device_train_batch_size = batch_size,   # batch size per device during training
    per_device_eval_batch_size = batch_size,    # batch size for evaluation
    output_dir = checkpoints_path,    # output directory
    overwrite_output_dir = False,
    load_best_model_at_end = False,
    predict_with_generate = True,
    generation_num_beams = 3, 
    eval_accumulation_steps = 2000,  # send logits and labels to cpu for evaluation step by step, rather than all together
    evaluation_strategy = 'steps',
    save_strategy = 'steps',
    save_total_limit = 3,   # Only last [save_total_limit] models are saved. Older ones are deleted.
    save_steps = 1000,
    eval_steps = 4000,    # Evaluation and Save happens every [eval_steps] steps
    learning_rate = 3e-5,
    num_train_epochs = 1,    # total number of training epochs
    warmup_steps = 500,   # number of warmup steps for learning rate scheduler
    weight_decay = 0.01    # strength of weight decay
)

trainer = CustomLossTrainer(
    compute_metrics = compute_metrics,
    generation_function = generate_caption,
    tokenizer = None,
    data_collator = None,
    model = model, # the instantiated 🤗 Transformers model to be trained
    args = training_args,   # training arguments, defined above
    train_dataset = data_train,   # training dataset
    eval_dataset = data_val   # evaluation dataset
)

trainer.train(checkpoint)
writer.flush()