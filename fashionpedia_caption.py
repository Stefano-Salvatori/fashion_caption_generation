import dataclasses
from typing import Any, List, Mapping, Union
import torch
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from transformers.deepspeed import is_deepspeed_zero3_enabled
from modules.train_utils import GenerationConfig, ModelComponents, init_model
from PIL import Image, ImageDraw, ImageChops
import os
from tqdm import tqdm
import json
import pycocotools.mask as maskUtils
import argparse

parser = argparse.ArgumentParser(description="Task Argument Parser")
parser.add_argument(
    "--caption_segmentations",
    action=argparse.BooleanOptionalAction,
    help="If true, generates a caption for each image segmentation, otherwise the caption is generated for the whole image.",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/home/salvatori/datasets/FashionPedia",
    required=False,
    help="Path to FashionPedia dataset",
)
parser.add_argument("--dataset_split", type=str, default="train", help="Dtaset split: train | test")
parser.add_argument(
    "--model_checkpoint",
    type=str,
    default="./checkpoints/entropy_triplet_hard_10epoch/checkpoint-113967",
    required=False,
    help="Model checkpoint path",
)
parser.add_argument("--batch_size", type=int, default=64, required=False, help="Batch size")


class FashionPediaImagesDataset(torch.utils.data.Dataset):
    DEFAULT_CLOTHES_CATEGORIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19, 21, 22, 23, 24, 25]

    def __init__(
        self,
        dataset_path: str,
        split: str,
        img_processor,
        caption_segmentations: bool,
        category_list: List[int] = DEFAULT_CLOTHES_CATEGORIES,
        area_threshold: int = 3000,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        annotations_file = (
            "instances_attributes_val2020.json" if split == "test" else "instances_attributes_train2020.json"
        )
        annotations_path = os.path.join(dataset_path, annotations_file)
        with open(annotations_path, "r") as j:
            self.data_annotations = json.loads(j.read())
            self.images = self.data_annotations["images"]
            # Keep only annotations of specific categories and with a sufficiently large area
            self.data_annotations = [
                a
                for a in self.data_annotations["annotations"]
                if a["category_id"] in category_list and a["area"] >= area_threshold
            ]
        self.img_processor = img_processor
        self.img_ids_to_metadata = {m["id"]: m for m in self.images}
        self.caption_segmentations = caption_segmentations

    def __getitem__(self, idx: int) -> dict:
        if not self.caption_segmentations:
            image_metadata = self.images[idx]
            img_path = self.__get_image_path(image_metadata["file_name"])
            image = Image.open(img_path).convert("RGB")
            return {
                "pixel_values": self.img_processor(image, return_tensors="pt")["pixel_values"][0],
                "img_id": image_metadata["id"],
            }
        else:
            annotation = self.data_annotations[idx]
            image_metadata = self.img_ids_to_metadata[annotation["image_id"]]
            img_path = self.__get_image_path(image_metadata["file_name"])
            image = Image.open(img_path).convert("RGB")
            image = self.__mask_image(image, annotation["segmentation"])
            image = self.__crop_to_rectangle(image, tuple(annotation["bbox"]))
            return {
                "pixel_values": self.img_processor(image, return_tensors="pt")["pixel_values"][0],
                "img_id": annotation["image_id"],
                "annotation_id": annotation["id"],
            }

    def __len__(self):
        return len(self.data_annotations) if self.caption_segmentations else len(self.images)

    def __mask_image(self, image: Image, segmentation: Union[list, dict]) -> Image:
        if isinstance(segmentation, dict):  # RLE annotation (doesn't have polygons)
            maskIm = Image.fromarray(maskUtils.decode(segmentation))
        else:
            # Select the clothes using its segmentation
            maskIm = Image.new("1", image.size, 1)
            imgDraw = ImageDraw.Draw(maskIm)
            for polygon in segmentation:
                imgDraw.polygon(polygon, fill=0)

        return ImageChops.add(image, maskIm.convert("RGB"))

    def __crop_to_rectangle(self, image, rectangle: tuple[float, float, float, float]):
        x, y, w, h = rectangle
        return image.crop((x, y, x + w, y + h))

    def __get_image_path(self, filename: str):
        return os.path.join(self.dataset_path, self.split, filename)


def _prepare_input(data: Union[torch.Tensor, Any], device: torch.device) -> Union[torch.Tensor, Any]:
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data


args = parser.parse_args()
caption_segmentations = args.caption_segmentations
dataset_path = args.dataset_path
split = args.dataset_split  # "train" or "test"
out_filename = f"{split}_annotations_captions.json" if caption_segmentations else f"{split}_captions.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = args.model_checkpoint  # "./checkpoints/entropy_triplet_hard_10epoch/checkpoint-113967"

generation_config = GenerationConfig(
    max_length=128,
    min_length=0,
    do_sample=False,
    num_beams=3,
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

# component configurations
encoder_decoder_components = ModelComponents(
    encoder_checkpoint="google/vit-base-patch16-224-in21k",
    decoder_checkpoint="gpt2",
    img_processor=ViTFeatureExtractor,
    generation_config=generation_config,
)

model = init_model(encoder_decoder_components, checkpoint)
model = model.to(device)
model.eval()

dataset = FashionPediaImagesDataset(
    dataset_path=dataset_path,
    split=split,
    img_processor=encoder_decoder_components.img_processor,
    caption_segmentations=caption_segmentations,
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


gen_kwargs = dataclasses.asdict(generation_config)
gen_kwargs = gen_kwargs | {
    "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
    "decoder_start_token_id": model.config.decoder_start_token_id,
    "eos_token_id": model.config.decoder.eos_token_id,
}
captions = {}
for j, batch in enumerate(tqdm(dataloader)):
    img_ids = batch.pop("img_id").tolist()
    annotation_ids = batch.pop("annotation_id").tolist() if "annotation_id" in batch else None
    input = _prepare_input(batch, device)
    output = model.generate(input[model.encoder.main_input_name], **gen_kwargs)
    output = encoder_decoder_components.tokenizer.batch_decode(output, skip_special_tokens=True)
    for i, img_id in enumerate(img_ids):
        key = img_id if annotation_ids is None else f"{img_id}_{annotation_ids[i]}"
        captions[key] = output[i]


with open(out_filename, "w") as outfile:
    json.dump(captions, outfile)
