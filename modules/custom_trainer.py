import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from packaging import version
from torch import nn
from transformers import Pipeline, PreTrainedTokenizer, Seq2SeqTrainer, pipeline
from transformers.integrations import TensorBoardCallback
from transformers.utils import logging
from modules.train_utils import GenerationConfig, LossType
from transformers.deepspeed import is_deepspeed_zero3_enabled
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


logger = logging.get_logger(__name__)


class CustomTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        generation_config: GenerationConfig,
        triplet_margin: float = 0.1,
        triplet_text_embedder: str = None,
        loss_type: LossType = LossType.ENTROPY,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.tokenizer = tokenizer
        # self.generation_function = generation_function if generation_function else self.model.generate
        self.generation_config = generation_config

        self.text_feature_extractor = self._pipeline_from_embedder(triplet_text_embedder, self.model.device.index)
        self.miner_func = miners.TripletMarginMiner(margin=triplet_margin, type_of_triplets="semihard")
        self.loss_func = losses.TripletMarginLoss(margin=triplet_margin, swap=True)
        # move Tensorboard Callback to last position so it can log all log items added in other callbacks
        for i, cb in enumerate(self.callback_handler.callbacks):
            if isinstance(cb, TensorBoardCallback):
                break
        self.callback_handler.callbacks.append(self.callback_handler.callbacks.pop(i))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        negative = inputs.pop("negative_pixel_values", None)
        positive_labels = inputs.pop("positive_label", None)
        negative_labels = inputs.pop("negative_label", None)
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            entropy_loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            entropy_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # TODO: change log call
            # self.log({"entropy_loss": torch.mean(entropy_loss).item()})
        if self.loss_type == LossType.ENTROPY_TRIPLET:
            positive_img_embedding = model.encoder(pixel_values=inputs["pixel_values"])["pooler_output"]
            negative_img_embedding = model.encoder(pixel_values=negative)["pooler_output"]
            caption = model.generate(
                inputs["pixel_values"],
                decoder_start_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
            )
            caption = self.tokenizer.batch_decode(caption, skip_special_tokens=True)
            # TODO: probably it is better to create a custom pipeline that returns tensors instead of lists
            # Take CLS embedding for each generated caption
            caption_embedding = torch.tensor(
                self.text_feature_extractor(caption, batch_size=len(caption)), device=self.model.device
            )[:, 0, 0, :]
            # concat and normalize embeddings
            embeddings = torch.cat(
                (
                    torch.nn.functional.normalize(caption_embedding),
                    torch.nn.functional.normalize(positive_img_embedding),
                    torch.nn.functional.normalize(negative_img_embedding),
                )
            )
            labels = torch.cat((positive_labels, positive_labels, negative_labels))
            hard_pairs = self.miner_func(embeddings, labels)
            triplet_loss = self.loss_func(embeddings, labels, hard_pairs)
            final_loss = entropy_loss + triplet_loss
            self.log(
                {"triplet_loss": torch.mean(triplet_loss).item(), "entropy_loss": torch.mean(entropy_loss).item()}
            )
        else:
            final_loss = entropy_loss
        return (final_loss, outputs) if return_outputs else final_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = ["negative_pixel_values"],
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = dataclasses.asdict(self.generation_config)
        gen_kwargs = gen_kwargs | {
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "decoder_start_token_id": self.model.config.decoder_start_token_id,
            "eos_token_id": self.model.config.decoder.eos_token_id,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(generation_inputs, **gen_kwargs)
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        input_copy = inputs.copy()
        for pop_key in ["negative_pixel_values", "positive_label", "negative_label"]:
            input_copy.pop(pop_key, None)

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**input_copy)
            else:
                outputs = model(**input_copy)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _pipeline_from_embedder(self, embedder_weights: str, device_idx: int = -1) -> Pipeline:
        return pipeline("feature-extraction", model=embedder_weights, device=device_idx)
