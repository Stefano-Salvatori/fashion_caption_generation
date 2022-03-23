# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classes to support Vision-Encoder-Text-Decoder architectures"""


from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss, TripletMarginLoss
from transformers import Pipeline, PreTrainedTokenizer, VisionEncoderDecoderModel, pipeline

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


logger = logging.get_logger(__name__)


@dataclass
class Seq2SeqLMOutputWithContrastive(Seq2SeqLMOutput):
    entropy_loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None


class VisionEncoderDecoderModelWithContrastive(VisionEncoderDecoderModel):
    def __init__(
        self,
        embedder_model_name_or_path: str,
        decoder_tokenizer: PreTrainedTokenizer,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config, encoder, decoder)
        self.embedder_model_name_or_path = embedder_model_name_or_path
        self.text_feature_extractor = self._pipeline_from_embedder(embedder_model_name_or_path)
        self.decoder_tokenizer = decoder_tokenizer

    def forward(
        self,
        pixel_values=None,
        negative_pixel_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        negative_encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        encoder_hidden_states, encoder_outputs = self._encode_image(
            pixel_values, encoder_outputs, output_attentions, output_hidden_states, return_dict, kwargs_encoder
        )
        encoder_attention_mask = None

        if negative_pixel_values is not None:
            negative_hidden_states, negative_encoder_outputs = self._encode_image(
                negative_pixel_values,
                negative_encoder_outputs,
                output_attentions,
                output_hidden_states,
                return_dict,
                kwargs_encoder,
            )

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        positive_image_embeddings = encoder_hidden_states[:, 0, :]
        negative_image_embeddings = negative_hidden_states[:, 0, :]

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        ce_loss = None
        contrastive_loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            # Cross entropy loss
            loss_fct = CrossEntropyLoss()
            ce_loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))
            
            if negative_pixel_values:
                # Triplet loss
                caption = self.decoder_tokenizer.batch_decode(torch.argmax(logits, dim=2))

                # TODO: probably it is better to create a custom pipeline that returns tensors instead of lists
                caption_embedding = torch.tensor(
                    self.text_feature_extractor(caption, batch_size=len(caption)), device=self.device
                )
                # Take CLS embedding for each generated caption
                caption_embedding = caption_embedding[:, 0, 0, :]
                # TODO: add margin hyperparameter
                contrastive_loss = TripletMarginLoss()
                contrastive_loss = contrastive_loss(caption_embedding, positive_image_embeddings, negative_image_embeddings)

            loss = ce_loss + contrastive_loss

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutputWithContrastive(
            loss=loss,
            entropy_loss=ce_loss,
            contrastive_loss=contrastive_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _encode_image(
        self, pixel_values, encoder_outputs, output_attentions, output_hidden_states, return_dict, kwargs_encoder
    ):
        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        return encoder_hidden_states, encoder_outputs

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        embedder_model_name_or_path: str,
        decoder_tokenizer: PreTrainedTokenizer,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path, decoder_pretrained_model_name_or_path, *model_args, **kwargs
        )
        return cls(
            embedder_model_name_or_path=embedder_model_name_or_path,
            decoder_tokenizer=decoder_tokenizer,
            encoder=model.encoder,
            decoder=model.decoder,
            config=model.config,
        )

    def to(self, *args, **kwargs):
        to_return = super().to(*args, **kwargs)
        self.text_feature_extractor = self._pipeline_from_embedder(self.embedder_model_name_or_path, self.device.index)
        return to_return

    def _pipeline_from_embedder(self, embedder_weights: str, device_idx: int = -1) -> Pipeline:
        return pipeline("feature-extraction", model=embedder_weights, device=device_idx)
