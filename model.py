import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from dataclasses import dataclass
import inspect

def gelu(x):
    #new, fast gelu function. See paper here: https://arxiv.org/abs/1606.08415
    return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.attension = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.flash:
            print("Running with fast attention")
        else:
            print("WARNING: Running with slow attension!!! (probably a problem with the pytorch version)")

    def forward(self, x):
        batch_size, sequence_len, embedding_dim = x.size()

        #get the query, key, and value
        query, key, value = self.attension(x).split(self.n_embd, dim=2)
        key = key.view(batch_size, sequence_len, self.n_head, embedding_dim // self.n_head).transpose(1,2)
        query = query.view(batch_size, sequence_len, self.n_head, embedding_dim // self.n_head).transpose(1,2)
        value = value.view(batch_size, sequence_len, self.n_head, embedding_dim // self.n_head).transpose(1,2)

        if self.flash:
            #quick self-attension
            y = F.scaled_dot_product_attention(query, key, value, attn_mask = None, dropout_p = self.dropout if self.training else 0, is_causal=True)
        else:
            #manual and slow self-attension
            attension = (query @ key.transpose(-2, -1)) * (1.0 / np.sqrt(key.size(-1)))
            attension = self.masked_fill(self.bias[:,:,:sequence_len, :sequence_len] == 0, float('-inf'))
            attension = F.softmax(att, dim=-1)
            attension = self.attn_dropout(attension)
            y = attension @ value
        
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_len, embedding_dim)
        return self.resid_dropout(self.projection(y))

class MultiLayerPerceptron(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.feed_fwd = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.feed_fwd(x)
        x = gelu(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    #Need to manually make a layer norm so that I can turn the biases on and off

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attension = SelfAttention(config)
        self.norm2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MultiLayerPerceptron(config)

    def forward(self, x):
        x+= self.attension(self.norm1(x))
        return x + self.mlp(self.norm2(x))

@dataclass
class GPTConfig:
    #GPT2 configuration
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True 

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            dropout = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm_func = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        #initialize all the weights (acording to GPT3 paper)
        self.apply(self._init_weights)

        for param_name, param in self.named_parameters():
            if param_name.endswith('projection.weight'):
                torch.nn.init.normal_(param, mean=0, std=0.02/np.sqrt(2*config.n_layer))

        #get the number of parameters
        n_params = sum(param.numel() for p in self.parameters())

        print("Initialization complete")
        print("Total number of parameters: ", n_params)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, sequence_len = idx.size()
        assert sequence_len <= self.config.block_size, "ERROR: Cannot create a forward sequence longer than the configured block size"
        position = torch.arange(0, sequence_len, dtype=torch.long, device=idx.device).unsqueeze(0)

        token_embedding = self.transformer.wte(idx)
        position_embedding = self.transformer.wpe(position)
        x = self.transformer.dropout(token_embedding + position_embedding)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.norm_func(x)

        #calculate the loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        #cool function that lets you reduce the block size even after the model has been trained
        #first mensioned in GPT2 paper, but enhanced with GPT 3 paper.

        assert block_size <= self.config.block_size, "ERROR: Cannot crop to a block size smaller than the configured block size"
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.blocks:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # ability to load in pretrained models from hundl (it's all pretty much just copy-pasted)
        assert model_type in  {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        #The parameters need to be seperated into ones that correlate with weight decay and ones that don't
        #Unfortunately, I didn't think about needing to do this until I coded everything esle, so the implmentation is sloppy
        #TODO: make this cleaner

        experience_decay = set()
        dont_experience_decay = set()
        cant_decay = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        definitely_decay = (torch.nn.Linear, )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                if module_name:
                    full_name = '%s.%s' % (module_name, param_name)
                else:
                    full_name = param_name

                if param_name.endswith('bias') or (param_name.endswith('weight') and isinstance(module, cant_decay)):
                    dont_experience_decay.add(full_name)
                if (param_name.endswith('weight') and isinstance(module, definitely_decay)):
                    experience_decay.add(full_name)
        experience_decay.remove('lm_head.weight')

        #check that the sets contain every set
        parameters = {param_name: param for param_name, param in self.named_parameters()}
        assert len(experience_decay & dont_experience_decay) == 0, "WARNING: parameter %s has been categorized as decaying and not-decaying" % (str(decay & no_decay))
        assert len(parameters.keys() - (experience_decay | dont_experience_decay)) == 0, "WARNING: parameter %s has not been categorized as decaying or not decaying"  % (str(parameters.keys() - (decay | no_decay)))

        #create optimizers
        decay_optimizer = {"params": [parameters[param_name] for param_name in sorted(list(experience_decay))], "weight_decay": weight_decay}
        dont_decay_optimizer = {"params": [parameters[param_name] for param_name in sorted(list(dont_experience_decay))], "weight_decay": 0}

        #Copy and pasted from the new fused option for AdamW from torch docs. Supposed to be much faster. 
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW([decay_optimizer, dont_decay_optimizer], lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        #Actually generates strings given inputs.

        for i in range(max_new_tokens):
            token_index = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, loss = self(token_index)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probabilities = F.softmax(logits, dim=1)
            next_token_index = torch.multinomial(probabilities, num_samples=1)
            token_index = torch.cat((token_index, next_token_index), dim=1)

        return token_index

