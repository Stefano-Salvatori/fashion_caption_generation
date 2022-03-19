from dataclasses import dataclass, fields
from typing import List
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PretrainedConfig, VisionEncoderDecoderModel
from transformers.feature_extraction_utils import FeatureExtractionMixin


class ModelComponents:
    def __init__(
        self,
        encoder_checkpoint: str,
        decoder_checkpoint: str,
        img_processor: FeatureExtractionMixin,
        generation_config: PretrainedConfig,
    ):
        self.encoder_checkpoint = encoder_checkpoint
        self.decoder_checkpoint = decoder_checkpoint
        self.generation_config = generation_config
        self.img_processor = img_processor.from_pretrained(self.encoder_checkpoint)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
        # tokenizer.padding_side = "left"  # TODO: why?
        if not self.tokenizer.pad_token:
            # we can use the  EOS token as PAD token if the tokenizer doesn't have one
            # (https://huggingface.co/docs/transformers/master/model_doc/vision-encoder-decoder#:~:text=model.config.pad_token_id%20%3D%20model.config.eos_token_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token


@dataclass
class GenerationConfig:
    max_length: int = 64
    min_length: int = 0
    do_sample: bool = False
    num_beams: int = 3
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    diversity_penalty: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    bad_words_ids: List[int] = None


def init_model(components_config: ModelComponents, checkpoint: str = None) -> VisionEncoderDecoderModel:
    """
    Initialize Vision Encoder-Decoder Model
    """
    # load models and their configs from pretrained checkpoints
    if checkpoint is None:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            components_config.encoder_checkpoint, components_config.decoder_checkpoint
        )
    else:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(checkpoint)

    model.config.decoder_is_decoder = True
    model.config.decoder_add_cross_attention = True
    model.config.decoder_start_token_id = (
        components_config.tokenizer.bos_token_id
        if components_config.tokenizer.bos_token_id is not None
        else components_config.tokenizer.cls_token_id
    )

    # we can use the  EOS token as PAD token if the tokenizer doesn't have one
    # (https://huggingface.co/docs/transformers/master/model_doc/vision-encoder-decoder#:~:text=model.config.pad_token_id%20%3D%20model.config.eos_token_id)
    model.config.pad_token_id = (
        components_config.tokenizer.pad_token_id
        if components_config.tokenizer.pad_token_id is not None
        else components_config.tokenizer.eos_token_id
    )
    model.config.decoder_bos_token_id = model.config.decoder_start_token_id
    model.config.decoder_eos_token_id = components_config.tokenizer.eos_token_id

    # set generation arguments
    for field in fields(components_config.generation_config):
        setattr(model.config.decoder, field.name, getattr(components_config.generation_config, field.name))
    
    return model


def generate_caption(
    model,
    tokenizer,
    pixel_values,
    num_beams: int = 3,
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


def normalize_0_to_1(x: torch.Tensor):
    # x -= x.min(1, keepdim=True)[0]
    # x /= x.max(1, keepdim=True)[0]
    # writer.add_text('tensor_shape', str(list(x.shape)))
    return torch.nn.functional.normalize(x, dim=1)


def get_bert_embedding(model, tokenizer, text, max_text_len: int, normalize: bool = True):
    # TODO: replace with huggingface pipeline
    text = tokenizer.batch_decode(text, skip_special_tokens=True)
    input_ids = tokenizer(
        text, truncation=True, max_length=max_text_len, padding="max_length", return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        embedding = model(**input_ids)["pooler_output"]
    return normalize_0_to_1(embedding) if normalize else embedding


def get_encoder_embedding(model, pixel_values, normalize: bool = True):
    embedding = model.encoder(pixel_values.to(model.device))["pooler_output"]
    return normalize_0_to_1(embedding) if normalize else embedding


def triplet_margin_loss(
    model,
    tokenizer,
    text_embedder_model,
    text_embedder_tokenizer,
    pixel_values,
    negatives,
    max_text_embedding_len: int = 64,
    margin: float = 0.1,
    swap: bool = True,
):
    negative_embeddings = get_encoder_embedding(model, negatives)
    positive_embeddings = get_encoder_embedding(model, pixel_values)
    captions = generate_caption(model, tokenizer, pixel_values)
    caption_embeddings = get_bert_embedding(
        text_embedder_model, text_embedder_tokenizer, captions, max_text_embedding_len
    )
    return torch.nn.functional.triplet_margin_loss(
        anchor=caption_embeddings, positive=positive_embeddings, negative=negative_embeddings, margin=margin, swap=swap
    )
