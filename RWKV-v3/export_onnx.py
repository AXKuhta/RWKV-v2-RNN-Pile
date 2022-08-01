########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import types
import copy
import torch
from torch.nn import functional as F

from src.model import RWKV_RNN
from src.model import RWKV_GPT

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# ---> Edit src/model.py to set MODEL_NAME and CPU / CUDA mode <---

context = '\nIn a shocking finding,'

##############################################################################################################

def sample_logits(out, temperature=1.0, top_p=0.7):
    probs = F.softmax(torch.tensor(out), dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)

    return torch.multinomial(probs, num_samples=1)[0]

def gpt_test(context):
	model = RWKV_GPT()
	model.eval()

	ctx = model.tokenizer.encode(context)

	for i in range(64):
		with torch.no_grad():
			x = model( torch.tensor([ctx]) )

		char = sample_logits( x[0][len(ctx) - 1].tolist() )
		char = char.item()

		print(model.tokenizer.decode(char), end='', flush=True)
		ctx.append(char)

def jit_test(context):
	model = RWKV_GPT()
	model.eval()

	ctx = model.tokenizer.encode(context)

	jit = torch.jit.script(model)

	for i in range(64):
		x = jit( torch.tensor([ctx]) )

		char = sample_logits( x[0][len(ctx) - 1].tolist() )
		char = char.item()

		print(model.tokenizer.decode(char), end='', flush=True)
		ctx.append(char)

def rnn_test(context):
	model = RWKV_RNN()

	xx_att = torch.zeros(12, 768)
	aa_att = torch.zeros(12, 768)
	bb_att = torch.zeros(12, 768)
	xx_ffn = torch.zeros(12, 768)

	ctx = model.tokenizer.encode(context)

	for i in range(64):
		ttx = [] + ctx

		while len(ttx) < 768:
			ttx.insert(0, 0)

		x, xx_att, aa_att, bb_att, xx_ffn = model( torch.tensor(ttx), xx_att, aa_att, bb_att, xx_ffn )

		char = sample_logits( x.tolist() )
		char = char.item()

		print(model.tokenizer.decode(char), end='', flush=True)
		ctx.append(char)


def gpt_export():
	model = RWKV_GPT()

	ctx = torch.randint(5000, (1, 767)) + 100

	torch.onnx.export(model, ctx, "rwkv.onnx", input_names = ["idx"], output_names = ["x"], verbose=True)

def rnn_export():
	model = RWKV_RNN()

	ctx = torch.randint(5000, (768,) ) + 100
	xx_att = torch.zeros(12, 768)
	aa_att = torch.zeros(12, 768)
	bb_att = torch.zeros(12, 768)
	xx_ffn = torch.zeros(12, 768)

	torch.onnx.export(model, args=(ctx, xx_att, aa_att, bb_att, xx_ffn), f="rwkv.onnx", input_names = ["idx", "xx_att", "aa_att", "bb_att", "xx_ffn"], output_names = ["x", "xx_att_r", "aa_att_r", "bb_att_r", "xx_ffn_r"], verbose=True)

def jit_export():
	model = RWKV_GPT()

	ctx = torch.randint(5000, (1, 767)) + 100

	jit = torch.jit.script(model)

	torch.onnx.export(jit, ctx, "rwkv.onnx", input_names = ["idx"], output_names = ["x"], verbose=True)


