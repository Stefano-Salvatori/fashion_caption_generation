import torch


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
