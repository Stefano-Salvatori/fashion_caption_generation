import h5py
import numpy as np
import torch
import pandas as pd
from tabulate import tabulate
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

def generate_caption(model, pixel_values, num_beams:int=5, do_sample:bool=False, top_p:float=1.0, top_k:int=50, repetition_penalty:float=20.0, max_length:int=72, temperature:int=1.0):
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
    #preds = [pred.split(' ') for pred in preds]
    #labels = [[label.split(' ')] for label in labels]
    preds = [[pred.split(' ') for pred in predlist] for predlist in preds]
    labels = [[[label.split(' ')] for label in labelslist] for labelslist in labels]

    # bleu
    bleu = bleu_metric.compute(predictions=preds, references=labels)
    
    result = {"bleu": bleu["score"], "meteor": meteor['meteor']*100, "rougeL": rouge['rougeL'][1][2]*100}

    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# ## Triplet-Loss Test
# ### Model and Data setup
step = '48000-32562'
loss_t = "entropy"
checkpoint = checkpoints_path + loss_t + '-checkpoint-' + str(step)
model, tokenizer, data_train, data_val = init_model_and_data(vit_gpt2, checkpoint=checkpoint, n_train=-1, n_val=-1, subcategory=True)
model = model.to(device)

tokenizer = vit_gpt2.tokenizer.from_pretrained(vit_gpt2.decoder_checkpoint)

# ### Tensorboard Monitoring
tensorboard_path = drive_path+'tensorboard/'
log_path = tensorboard_path+"swap_from_scratch_subcat_altnorm_lowmargin"

writer = SummaryWriter(log_dir=log_path)

# ### Loss and Trainer config

def generate_predictions(model, do_sample:bool=False):
    predicted = list()
    for i in tqdm(range(0, len(data_val), batch_size)):
      batch = np.empty((batch_size, 3, 224, 224))
      l = 0
      for j in range(i, i+batch_size):
        batch[l] = data_val[j]['pixel_values']
        l+=1
      batch = torch.tensor(batch, device=device, dtype=torch.float)
      predicted.append(generate_caption(model, batch, do_sample=do_sample).cpu())
    predicted = np.asarray(predicted)
    # save to disk
    with open(drive_path + 'predictions/pred-' + loss_t + '-' + str(step) + '.npy', 'wb') as file:
      np.save(file, predicted)

def load_predictions(load_metrics:bool=False):
  with open(drive_path + 'predictions/pred-' + loss_t + '-' + str(step) + '.npy', 'rb') as file:
    predicted = np.load(file, allow_pickle=True)
  predicted = [item for sublist in predicted for item in sublist]
  #predicted = pd.Series(map(lambda cap: tokenizer.batch_decode(cap, skip_special_tokens=True), predicted))
  # REAL CAPTIONS
  cap_val = list()
  for p in tqdm(FashionGenDataset(data_path + "fashiongen_validation.h5").raw_h5()["input_description"], position=0, leave=True):
      cap_val.append(p[0].decode("ISO-8859-9")) #DEFUALT_STRINGS_ENCODING = "ISO-8859-9")
  cap_val = cap_val[0:len(predicted)]
  # CATEGORIES
  cat_val = list()
  for p in tqdm(FashionGenDataset(data_path + "fashiongen_validation.h5").raw_h5()["input_category"], position=0, leave=True):
      cat_val.append(p[0].decode("ISO-8859-9")) #DEFUALT_STRINGS_ENCODING = "ISO-8859-9")
  cat_val = pd.Series(cat_val[0:len(predicted)])
  # DATAFRAME
  data = pd.DataFrame({'caption':predicted, 'category':cat_val})
  # METRICS
  if(load_metrics):
    with open(drive_path + 'predictions/metrics-' + loss_t + '-' + str(step) + '.npy', 'rb') as file:
      scores = np.load(file, allow_pickle=True)
  else:
    scores = {}
    scores['bleu'] = []
    scores['rougeL'] = []
    scores['meteor'] = []
    scores['avg'] = []
    print("LEN:::"+str(len(data.caption.values)))
    print("SHAPE:::"+str(data.caption.values.shape))
    for i in tqdm(range(0, len(data.caption.values))):
      # print(data.caption.values[i])
      score = compute_metrics([torch.unsqueeze(data.caption.values[i], 0).cpu(), torch.unsqueeze(torch.tensor(tokenizer.encode(cap_val[i])), 0).cpu()], decode=True)
      # score = compute_metrics([[data.caption.values[i]], [cap_val[i]]], decode=False)
      scores['bleu'].append(score['bleu'])
      scores['meteor'].append(score['meteor'])
      scores['rougeL'].append(score['rougeL'])
      avg_score = sum(score.values()) / len(score)
      scores['avg'].append(avg_score)
    with open(drive_path + 'predictions/metrics-' + loss_t + '-' + str(step) + '.npy', 'wb') as file:
      np.save(file, scores)
  # RETURN FINAL DATASET
  data['score'] = scores
  data['real'] = cap_val
  return data

generate_predictions(model)
data = load_predictions()