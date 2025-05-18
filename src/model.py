from transformers import GPT2Config, GPT2LMHeadModel

def get_model(vocab_size):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=256,
        n_layer=6,
        n_head=8
    )
    return GPT2LMHeadModel(config)
