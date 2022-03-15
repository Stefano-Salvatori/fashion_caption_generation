import torch
from enum import Enum
from modules.fashiongen_utils import FashionGenDataset, Product


class NegativeSampleType(Enum):
    RANDOM = 1
    SAME_CATEGORY = 2
    SAME_SUBCATEGORY = 3


class FashionGenTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_name,
        caption_encodings,
        img_processor,
        n_samples: int,
        device: torch.device,
        negative_sample_type: NegativeSampleType = NegativeSampleType.SAME_SUBCATEGORY,
    ):
        self.n_samples = n_samples
        self.file_path = file_name
        self.dataset = FashionGenDataset(file_name)
        # h5py.File(file_name, mode="r")["input_image"]
        self.device = device

        self.caption_encodings = caption_encodings
        self.img_processor = img_processor
        self.negative_sample_type = negative_sample_type
        if self.n_samples == -1:
            self.n_samples = len(self.dataset)
        else:
            assert n_samples <= len(self.dataset), "n_samples must be <=" + str(len(self.dataset))

    def __getitem__(self, idx):
        product = self.dataset.get_product(idx)
        negative_product = self.get_negative(product)
        return {
            "pixel_values": self.preprocess_image(product.image),#.to(self.device),
            "labels": self.caption_encodings["input_ids"][idx],#.to(self.device),
            "attention_mask": self.caption_encodings["attention_mask"][idx],#.to(self.device),
            "negative": self.preprocess_image(negative_product.image),#.to(self.device),
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
