import torch
from enum import Enum

from transformers import PreTrainedTokenizer
from modules.data.fashiongen_utils import FashionGenDataset, Product


class NegativeSampleType(Enum):
    RANDOM = 1
    SAME_CATEGORY = 2
    SAME_SUBCATEGORY = 3


class FashionGenTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_name,
        text_tokenizer: PreTrainedTokenizer,
        img_processor,
        n_samples: int = -1,
        max_text_length: int = 256,
        sample_negative: bool = False,
        negative_sample_type: NegativeSampleType = NegativeSampleType.RANDOM,
        return_index: bool = False,
    ):
        self.n_samples = n_samples
        self.file_path = file_name
        self.dataset = FashionGenDataset(file_name)
        # h5py.File(file_name, mode="r")["input_image"]
        self.text_tokenizer = text_tokenizer

        # check if the EOS is equals to the PAD token
        self.eos_eq_pad = self.text_tokenizer.eos_token_id == self.text_tokenizer.pad_token_id

        self.max_text_length = max_text_length

        self.img_processor = img_processor
        self.negative_sample_type = negative_sample_type
        self.sample_negative = sample_negative
        self.return_index = return_index

        if self.n_samples == -1:
            self.n_samples = len(self.dataset)
        else:
            assert n_samples <= len(self.dataset), "n_samples must be <=" + str(len(self.dataset))

    def get_image(self, idx):
        return self.dataset.get_product(idx).image

    def __getitem__(self, idx):
        product = self.dataset.get_product(idx)
        # TODO: check if we have to manually add BOS and EOS (?)
        tokenized = self.text_tokenizer.encode_plus(
            self._clean_caption(product.decoded_caption()),
            max_length=self.max_text_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        labels = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        if self.eos_eq_pad:
            # We ignore all PAD tokens except for the first one that occurs (when it is the EOS token)
            attention_mask[attention_mask.sum()] = 1

        # make sure that PAD tokens are ignored by the loss function (set them to -100).
        labels = labels * attention_mask - 100 * (1 - attention_mask)
        to_return = {"pixel_values": self.preprocess_image(product.image), "labels": labels}

        if self.sample_negative:
            negative_product = self.get_negative(product)
            to_return["negative_pixel_values"] = (self.preprocess_image(negative_product.image),)

        if self.return_index:
            to_return["idx"] = idx

        return to_return

    def __len__(self):
        return self.n_samples

    def get_fashiongen(self) -> FashionGenDataset:
        return self.dataset

    def preprocess_image(self, image):
        return self.img_processor(image, return_tensors="pt")["pixel_values"][0]

    def get_negative(self, product: Product) -> Product:
        if self.negative_sample_type == NegativeSampleType.SAME_SUBCATEGORY:
            return self.dataset.get_same_subcategory_of(product)[0]
        elif self.negative_sample_type == NegativeSampleType.SAME_CATEGORY:
            return self.dataset.get_same_category_of(product)[0]
        elif self.negative_sample_type == NegativeSampleType.RANDOM:
            return self.dataset.get_random_product(product.p_id)
        else:
            raise ValueError(f"Negative sample type: {self.negative_sample_type}, not recognized")

    def _clean_caption(self, caption: str):
        return caption.strip().replace("<br>", " ")

