from transformers import PreTrainedTokenizerFast
from onnxruntime import InferenceSession
from torch.nn import functional as F
import numpy as np
import torch

def lprint(txt):
	print(txt, end='', flush=True)

def sample_logits(out, temperature=1.0, top_p=0.7):
	probs = F.softmax(torch.tensor(out), dim=-1)
	sorted_probs, _ = torch.sort(probs, descending=True)

	cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
	cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
	probs[probs < cutoff] = 0

	if temperature != 1.0:
		probs = probs.pow(1.0 / temperature)

	return torch.multinomial(probs, num_samples=1)[0]


tokenizer = PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")
session = InferenceSession("rwkv.onnx")

text = """\nIn a shocking finding,"""

ctx = tokenizer.encode(text)
print("Tokens in context:", len(ctx))

lprint( tokenizer.decode(ctx) )

for i in range(64):
	tgt = len(ctx)
	ttx = []

	for id in ctx:
		ttx.append(id)

	while len(ttx) < 767:
		ttx.append( 0 )

	inputs = { "idx": [ttx] }

	# [1][1][767][50277] GPT
	# [1][1][50277] RNN
	outputs = session.run(output_names=["x"], input_feed=inputs)
	state = outputs[0][0][tgt - 1]

	char = sample_logits(state)
	char = char.item()

	lprint( tokenizer.decode(char) )
	ctx.append(char)

