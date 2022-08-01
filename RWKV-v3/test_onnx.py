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

xx_att = torch.zeros(12, 768).tolist()
aa_att = torch.zeros(12, 768).tolist()
bb_att = torch.zeros(12, 768).tolist()
xx_ffn = torch.zeros(12, 768).tolist()

for i in range(64):
	tgt = len(ctx)
	ttx = []

	for id in ctx:
		ttx.append(id)

	# RNN takes the very last token
	# Pad the input from the front
	while len(ttx) < 768:
		ttx.insert(0, 0)

	inputs = { "idx": ttx, "xx_att": xx_att, "aa_att": aa_att, "bb_att": bb_att, "xx_ffn": xx_ffn }

	outputs = session.run(output_names=["x", "xx_att_r", "aa_att_r", "bb_att_r", "xx_ffn_r"], input_feed=inputs)
	state = outputs[0] # [50277]

	# [12][768]
	xx_att = outputs[1]
	aa_att = outputs[2]
	bb_att = outputs[3]
	xx_ffn = outputs[4]

	char = sample_logits(state)
	char = char.item()

	lprint( tokenizer.decode(char) )
	ctx.append(char)

