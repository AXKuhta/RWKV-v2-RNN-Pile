########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import math, json, time, types, copy, sys, os
import torch
from torch.nn import functional as F
import torch.nn as nn

from transformers import PreTrainedTokenizerFast

RUN_DEVICE = 'cpu' # cpu cuda
ctx_len = 768
n_layer = 12
n_embd = 768

# ---> download RWKV-3 169M model from https://huggingface.co/BlinkDL/rwkv-3-pile-169m/tree/main

MODEL_NAME = '20220720'

vocab_size = 50277
VOCAB_NAME = '20B_tokenizer.json'

print(f'\n* running on {RUN_DEVICE}')

################################################################################################################

class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, n_embd))
        
        hidden_sz = 4 * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_decay = nn.Parameter(torch.ones(n_embd, 1))
        self.time_curve = torch.tensor([-(ctx_len - 2 - i) for i in range(ctx_len-1)]).unsqueeze(0)
        self.time_first = nn.Parameter(torch.ones(n_embd, 1) * math.log(0.3))
        
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.time_mix_k = nn.Parameter(torch.ones(1,1,n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1,1,n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1,1,n_embd))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)

        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        #B, T, C = x.size()
        K_EPS = 1e-8
        B = 1
        T = 767
        C = 768

        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk).transpose(-1, -2)
        v = self.value(xv).transpose(-1, -2)
        r = self.receptance(xr)

        k = torch.clamp(k, max=60)
        k = torch.exp(k)

        kv = k * v

        time_w = torch.cat([torch.exp(self.time_decay) * self.time_curve.to(self.time_decay.device), self.time_first], dim=-1)
        w = torch.exp(time_w)
        
        w = w[:,-T:].unsqueeze(1)
        wkv = F.conv1d(torch.cat((torch.zeros(1, 768, T-1), kv), 2), w, groups=C)
        wk = F.conv1d(torch.cat((torch.zeros(1, 768, T-1), k), 2), w, groups=C) + K_EPS

        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)
        
        rwkv = self.output(rwkv)
        return rwkv

class Block(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        else:
            self.ln0 = torch.zeros
        
        self.att = RWKV_TimeMix(layer_id)
        self.ffn = RWKV_ChannelMix(layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class RWKV_GPT(nn.Module):
    def __init__(self, MODEL_NAME=MODEL_NAME):
        super().__init__()
        print('\nloading RWKV-GPT', MODEL_NAME)

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_NAME)
        self.emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.ctx_len = ctx_len
        #self.eval()
        self.load_state_dict(torch.load(MODEL_NAME + '.pth', map_location=torch.device('cpu')))
        #self.eval()

    def forward(self, idx):
        #B, T = idx.size()
        #assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."
        
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        x = self.head(x)

        return x

################################################################################################################

time_buf = {}

class RWKV_RNN():
    def __init__(self, MODEL_NAME=MODEL_NAME):
        print('\nloading RWKV-RNN', MODEL_NAME)
        self.ctx_len = ctx_len
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_NAME)
        self.training=False

        self.w = types.SimpleNamespace()
        
        w = torch.load(MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))

        for x in w.keys():
            if '.time_' in x:
                w[x] = w[x].squeeze()
            if '.time_decay' in x:
                w[x] = torch.exp(-torch.exp(w[x]))
            if '.time_first' in x:
                w[x] = torch.exp(w[x])
                    
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.clear()
    
    def clear(self):
        self.xx = {}
        self.aa = {}
        self.bb = {}
    def save(self, target):
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
    def load(self, target):
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)

    def LN(self, xx, w):
        return F.layer_norm(xx, (n_embd,), weight=w.weight, bias=w.bias)

    def FF(self, xx, w, name):
        #if name not in self.xx:
        #    self.xx[name] = torch.zeros(n_embd, device=RUN_DEVICE)

        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)

        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ xr)
        k = torch.square(torch.relu(w.key.weight @ xk))
        kv = w.value.weight @ k

        return r * kv

    def SA(self, xx, w, name):
        K_EPS = 1e-8

        #if name not in self.xx:
        #    self.xx[name] = torch.zeros(n_embd, device=RUN_DEVICE)
        #    self.aa[name] = torch.zeros(n_embd, device=RUN_DEVICE)
        #    self.bb[name] = torch.zeros(n_embd, device=RUN_DEVICE)

        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xv = xx * w.time_mix_v + self.xx[name] * (1 - w.time_mix_v)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)

        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ xr)

        k = torch.exp(torch.clamp(w.key.weight @ xk, max=60))
        v = w.value.weight @ xv
        kv = k * v

        a = self.aa[name] + w.time_first * kv
        b = self.bb[name] + w.time_first * k
        self.aa[name] = w.time_decay * self.aa[name] + kv
        self.bb[name] = w.time_decay * self.bb[name] + k

        rwkv = r * a / (b + K_EPS)

        return w.output.weight @ rwkv

    def run(self, ctx, xx_att, aa_att, bb_att, xx_ffn):
        w = self.w
        x = w.emb.weight[ctx[-1]]

        x = self.LN(x, w.blocks[0].ln0)
        for i in range(n_layer):
            self.xx[f'att.{i}'] = xx_att[i]
            self.aa[f'att.{i}'] = aa_att[i]
            self.bb[f'att.{i}'] = bb_att[i]
            self.xx[f'ffn.{i}'] = xx_ffn[i]
            x = x + self.SA(self.LN(x, w.blocks[i].ln1), w.blocks[i].att, f'att.{i}')
            x = x + self.FF(self.LN(x, w.blocks[i].ln2), w.blocks[i].ffn, f'ffn.{i}')

        x = self.LN(x, w.ln_out)

        x = w.head.weight @ x
        #x = x.tolist()

        xx_att_cd = []
        aa_att_cd = []
        bb_att_cd = []
        xx_ffn_cd = []

        for i in range(n_layer):
             xx_att_cd.append( self.xx[f'att.{i}'] )
             aa_att_cd.append( self.aa[f'att.{i}'] )
             bb_att_cd.append( self.bb[f'att.{i}'] )
             xx_ffn_cd.append( self.xx[f'ffn.{i}'] )

        xx_att_r = torch.stack(xx_att_cd)
        aa_att_r = torch.stack(aa_att_cd)
        bb_att_r = torch.stack(bb_att_cd)
        xx_ffn_r = torch.stack(xx_ffn_cd)

        return x, xx_att_r, aa_att_r, bb_att_r, xx_ffn_r

    def forward(self, ctx, xx_att, aa_att, bb_att, xx_ffn):
        return self.run(ctx, xx_att, aa_att, bb_att, xx_ffn)

    def __call__(self, ctx, xx_att, aa_att, bb_att, xx_ffn):
        return self.run(ctx, xx_att, aa_att, bb_att, xx_ffn)

    def train(self, x):
        pass

    def modules(self):
        return []

    def state_dict(self, keep_vars):
        return {}

################################################################################################################
