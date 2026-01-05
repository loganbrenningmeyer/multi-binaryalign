from transformers import AutoModel


def load_backbone(model_name: str, vocab_size: int) -> AutoModel:
    model = AutoModel.from_pretrained(model_name)

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    if model_vocab_size != vocab_size:
        model.resize_token_embeddings(vocab_size)

    return model