import torch
from enum import Enum

from transformers import PreTrainedTokenizer
from modules.data.fashiongen_utils import DEFAULT_STRINGS_ENCODING, FashionGenDataset, Product


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
        n_samples: int,
        max_text_length: int = 256,
        negative_sample_type: NegativeSampleType = NegativeSampleType.SAME_SUBCATEGORY,
    ):
        self.n_samples = n_samples
        self.file_path = file_name
        self.dataset = FashionGenDataset(file_name)
        # h5py.File(file_name, mode="r")["input_image"]
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length

        self.img_processor = img_processor
        self.negative_sample_type = negative_sample_type
        if self.n_samples == -1:
            self.n_samples = len(self.dataset)
        else:
            assert n_samples <= len(self.dataset), "n_samples must be <=" + str(len(self.dataset))

    def __getitem__(self, idx):
        product = self.dataset.get_product(idx)
        negative_product = self.get_negative(product)
        labels = self.text_tokenizer.encode_plus(
            self._clean_caption(product.caption.decode(DEFAULT_STRINGS_ENCODING)),
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
        ).input_ids

        # make sure that PAD tokens are ignored by the loss function
        # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb
        labels = [label if label != self.text_tokenizer.pad_token_id else -100 for label in labels]
        return {
            "pixel_values": self.preprocess_image(product.image),
            "labels": torch.tensor(labels),
            "negative_pixel_values": self.preprocess_image(negative_product.image),
        }

    def __len__(self):
        return self.n_samples

    def preprocess_image(self, image):
        return self.img_processor(image, return_tensors="pt")["pixel_values"][0]

    def get_negative(self, product: Product):
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

