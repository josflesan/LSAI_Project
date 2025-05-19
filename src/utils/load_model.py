from model.model import Transformer, TransformerModelArgs
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")

model_config = TransformerModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
    vocab_size=tokenizer.vocab_size,
    seq_len=4096,
)
model = Transformer(model_config)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
