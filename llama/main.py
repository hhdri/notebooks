from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
# from torchtune.models.llama3_2 import llama3_2_1b
from my_llama import llama3_2_1b
from torchtune.models.convert_weights import hf_to_tune
import torch
from safetensors import safe_open

with open ('input.txt', 'r') as f:
    text = f.read()
# the following is needed to make tokenization invertible
text = text.replace(' \'', '\'') # remove space before apostrophe
text = text[:100_000]

MODEL_SLUG = 'meta-llama/Llama-3.2-1B'
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_SLUG)
model = LlamaForCausalLM.from_pretrained(MODEL_SLUG)

model_path = '/Users/majid/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/model.safetensors'
model_tt = llama3_2_1b()

state_dict = {}
all_keys = []
with safe_open(model_path, framework="pt", device="cpu") as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)
        all_keys.append(k)
# dump all keys to a file
with open('all_keys.txt', 'w') as f:
    for k in all_keys:
        f.write(f'{k}\n')
state_dict = hf_to_tune(state_dict, num_heads=32, num_kv_heads=8, dim=2048, head_dim=64)
model_tt.load_state_dict(state_dict)
# model_tt = torch.compile(model_tt)

text_enc = tokenizer(text, return_tensors='pt')

batch_size = 4
context_length = 256
residual = text_enc['input_ids'].shape[1] % (batch_size * context_length)
print(text_enc['input_ids'].shape)
batches = text_enc['input_ids'][0, :-residual].reshape(-1, batch_size, context_length).to('cpu')
batches_attention_mask = torch.ones_like(batches).to('cpu')
print(batches.shape)

batch_idx = 22

with torch.no_grad():
    out = model_tt(batches[batch_idx])
    out2 = model(batches[batch_idx]).logits

labels = batches[batch_idx][:, 1:]
pred = out[:, :-1, :]
pred = torch.gather(pred, 2, labels.unsqueeze(-1)).squeeze(-1)
pred2 = out2[:, :-1, :]
pred2 = torch.gather(pred2, 2, labels.unsqueeze(-1)).squeeze(-1)

assert torch.allclose(pred, pred2)