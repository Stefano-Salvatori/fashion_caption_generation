import dataclasses
from typing import Any, Mapping, Union
import torch
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from transformers.deepspeed import is_deepspeed_zero3_enabled
from modules.data.dataset import FashionGenTorchDataset
from modules.train_utils import GenerationConfig, ModelComponents, init_model
from PIL import Image


def _prepare_input(data: Union[torch.Tensor, Any], device: torch.device) -> Union[torch.Tensor, Any]:
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
validation_dataset_path = "/datasets/FashionGen/fashiongen_validation.h5"

checkpoint = "./checkpoints/entropy_triplet_hard_10epoch/checkpoint-113967"

max_captions_length = 128
min_captions_length = 0
generation_config = GenerationConfig(
    max_length=max_captions_length,
    min_length=min_captions_length,
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


dataset = FashionGenTorchDataset(
    file_name=validation_dataset_path,
    text_tokenizer=encoder_decoder_components.tokenizer,
    img_processor=encoder_decoder_components.img_processor,
    sample_negative=False,
    max_text_length=max_captions_length,
    return_index=True,
)
fashiongen = dataset.get_fashiongen()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


gen_kwargs = dataclasses.asdict(generation_config)
gen_kwargs = gen_kwargs | {
    "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
    "decoder_start_token_id": model.config.decoder_start_token_id,
    "eos_token_id": model.config.decoder.eos_token_id,
}
for batch in dataloader:
    idx = batch.pop("idx")
    product = fashiongen.get_product(idx)
    img = Image.fromarray(product.image)
    input = _prepare_input(batch, device)
    output = model.generate(input[model.encoder.main_input_name], **gen_kwargs)
    output = encoder_decoder_components.tokenizer.batch_decode(output)
    print(output)
    print(f"REAL: {product.decoded_caption()}")
    img.save("img.png")
    break

