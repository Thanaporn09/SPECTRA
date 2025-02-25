from __future__ import absolute_import, division, print_function
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import modules.utils as utils

# External dependencies.
from mamba_ssm.modules.mamba_simple import Mamba
from .att_model_SPECTRA import pack_wrapper, AttModel

def transform_tokens2regions(hidden_states, num_regions, region_size):
    """
    Split a sequence [B, seq, D] into regions.
    Returns: [(B * num_regions), region_size, D]
    """
    return rearrange(hidden_states, 'b (nr rs) d -> (b nr) rs d', nr=num_regions, rs=region_size)

def transform_masks2regions(mask, num_regions, region_size):
    """
    Split mask [B, seq, 1] into regions.
    Returns: [(B * num_regions), region_size, 1]
    """
    return rearrange(mask, 'b (nr rs) 1 -> (b nr) rs 1', nr=num_regions, rs=region_size)


class RotaryPositionalEncoding(nn.Module):
    """
    Applies a rotary (cosine-sine) positional encoding.
    Input: [B, seq, D]
    """
    def __init__(self, d_model, dropout, max_len=70000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        angle_base = torch.linspace(0, 2 * np.pi, max_len)
        self.register_buffer('angle_base', angle_base)

    def forward(self, x):
        seq_len = x.size(1)
        angles = self.angle_base[:seq_len].unsqueeze(0).unsqueeze(-1)  # [1, seq, 1]
        rot_x = x * torch.cos(angles) + x.roll(1, dims=-1) * torch.sin(angles)
        return self.dropout(rot_x)

def pad_sequence_tokens(hidden_states, mask, region_size):
    """
    Pads hidden_states so that its sequence length becomes a multiple of (region_size - 1).
    hidden_states: [B, seq, D]
    mask: [B, seq, 1]
    Returns: padded hidden_states, padded mask, num_regions.
    """
    B, seq, D = hidden_states.size()
    if mask.size(1) == 1 and mask.size(2) == seq:
        mask = mask.transpose(1, 2)
    assert mask.shape == (B, seq, 1), f"Expected mask shape (B, seq, 1) but got {mask.shape}"
    num_regions = int(np.ceil(seq / (region_size - 1)))
    total_tokens = num_regions * (region_size - 1)
    padding_size = total_tokens - seq
    if padding_size > 0:
        hidden_padding = torch.zeros(B, padding_size, D, device=hidden_states.device)
        hidden_states = torch.cat([hidden_states, hidden_padding], dim=1)
        mask_padding = torch.zeros(B, padding_size, 1, device=mask.device)
        mask = torch.cat([mask, mask_padding], dim=1)
    assert hidden_states.size(1) == num_regions * (region_size - 1), "Padding mismatch in hidden_states"
    return hidden_states, mask, num_regions

def insert_global_tokens(hidden_states, mask, global_token, region_size, num_regions):
    """
    Splits padded hidden_states into regions, appends a global token to each region,
    and recombines them.
    """
    B, total_tokens, D = hidden_states.size()
    hidden_states = hidden_states.view(B, num_regions, region_size - 1, D)
    global_tokens = repeat(global_token, '1 1 d -> b nr 1 d', b=B, nr=num_regions)
    hidden_states = torch.cat([hidden_states, global_tokens], dim=2)  # [B, num_regions, region_size, D]
    hidden_states = rearrange(hidden_states, 'b nr rs d -> b (nr rs) d')
    
    mask = rearrange(mask, 'b (nr rs) 1 -> b nr rs 1', nr=num_regions, rs=region_size - 1)
    global_mask = torch.ones(B, num_regions, 1, 1, device=mask.device)
    mask = torch.cat([mask, global_mask], dim=2)
    mask = rearrange(mask, 'b nr rs 1 -> b (nr rs) 1')
    
    # Assert that the new length equals num_regions * region_size.
    assert hidden_states.size(1) == num_regions * region_size, "Global token insertion dimension mismatch"
    return hidden_states, mask

class EncoderLayer(nn.Module):
    def __init__(self, layer_idx, heads=8, d_model=512, d_ff=512, region_size=256,
                 use_region_encoder=True, use_WSI_encoder=False, dropout=0.1,
                 max_patch=100000, first_layer=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.region_size = region_size
        self.max_patch = max_patch
        self.first_layer = first_layer
        self.d_model = d_model

        d_state = min(16 * (layer_idx + 1), 128)
        expand = 2 if layer_idx < 2 else 4

        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Use rotary positional encoding
        self.region_position_embeddings = RotaryPositionalEncoding(d_model, dropout)

        self.use_region_encoder = use_region_encoder
        self.use_WSI_encoder = use_WSI_encoder

        if self.use_region_encoder:
            self.region_encoder = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
        if self.use_WSI_encoder:
            self.WSI_encoder = Mamba(d_model=d_model, d_state=d_state * 2, d_conv=4, expand=expand)

    def forward(self, x, mask, num_regions):
        # x: [B, seq, D], mask: [B, seq, 1]
        if self.first_layer:
            x, mask = self.interpolate_global_token(x, mask)
        if self.use_region_encoder:
            region_inputs = transform_tokens2regions(x, num_regions, self.region_size)
            region_masks = transform_masks2regions(mask, num_regions, self.region_size)
            region_inputs = self.region_position_embeddings(region_inputs)
            region_inputs = region_inputs * region_masks
            outputs = self.region_encoder(region_inputs)
            outputs = rearrange(outputs, '(b nr) rs d -> b (nr rs) d', nr=num_regions, rs=self.region_size)
        else:
            outputs = x

        if self.use_WSI_encoder:
            # Extract one token per region and process via WSI branch.
            global_tokens = outputs[:, ::self.region_size]
            global_tokens = self.region_position_embeddings(global_tokens)
            WSI_outputs = self.WSI_encoder(global_tokens)
            gate = torch.ones_like(global_tokens)  # or use a learned gating
            outputs[:, ::self.region_size] = global_tokens * gate + WSI_outputs * (1 - gate)
        return outputs, mask

    def interpolate_global_token(self, hidden_states, mask):
        hidden_states, mask, num_regions = pad_sequence_tokens(hidden_states, mask, self.region_size)
        hidden_states, mask = insert_global_tokens(hidden_states, mask, self.global_token, self.region_size, num_regions)
        return hidden_states, mask

class AttentivePooling(nn.Module):
    def __init__(self, encoder_layout):
        super().__init__()
        self.lin_proj = nn.Linear(encoder_layout['d_model'], encoder_layout['d_model'])
        self.v = nn.Linear(encoder_layout['d_model'], 1, bias=False)
        self.dropout = encoder_layout['dropout']

    def forward(self, inputs):
        # inputs: [B, num_regions, region_size, D]
        lin_out = self.lin_proj(inputs)
        attn_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
        attn_norm = F.softmax(attn_weights, dim=-1)
        pooled = torch.sum(attn_norm.unsqueeze(-1) * inputs, dim=2)
        return F.dropout(pooled, p=self.dropout, training=self.training)

class EncoderPooler(nn.Module):
    def __init__(self, encoder_layout, pooling='attentive'):
        super().__init__()
        self.dense = nn.Linear(encoder_layout['d_model'], encoder_layout['d_model'])
        self.pooling = pooling
        if self.pooling == 'attentive':
            self.attentive_pooling = AttentivePooling(encoder_layout)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states: [B, num_regions, region_size, D] or flattened [B, seq, D]
        if self.pooling == 'attentive':
            pooled = self.attentive_pooling(hidden_states)
        else:
            pooled = torch.max(hidden_states, dim=2)[0]
        pooled = self.dense(pooled)
        return self.activation(pooled)

class Encoder(nn.Module):
    def __init__(self, encoder_layout):
        super().__init__()
        self.encoder_layout = encoder_layout
        self.layer = nn.ModuleList([
            EncoderLayer(
                layer_idx=idx,
                heads=encoder_layout['num_heads'],
                d_model=encoder_layout['d_model'],
                d_ff=encoder_layout['d_ff'],
                region_size=encoder_layout['region_size'],
                use_region_encoder=encoder_layout[str(idx)]['region_encoder'],
                use_WSI_encoder=encoder_layout[str(idx)]['WSI_encoder'],
                dropout=encoder_layout['dropout'],
                first_layer=encoder_layout[str(idx)]['first_layer']
            ) for idx in range(int(encoder_layout['num_layers']))
        ])
        self.norm = LayerNorm(encoder_layout['d_model'])
        self.pooler = EncoderPooler(encoder_layout, pooling=encoder_layout['pooling'])
        self.region_size = encoder_layout['region_size']

    def forward(self, x, mask):
        # x: [B, seq_len, d_model], mask: [B, seq_len, 1] or [B, 1, seq_len]
        B, seq_len, d_model = x.shape
        
        # If the sequence is too short for region grouping, skip region-based processing.
        if seq_len < self.region_size - 1:
            x = self.norm(x)
            if mask.dim() == 3 and mask.size(1) == seq_len:
                mask = mask.transpose(1, 2)
            return x, mask

        num_regions = int(np.ceil(seq_len / (self.region_size - 1)))
        for layer in self.layer:
            x, mask = layer(x, mask, num_regions)

        x = self.norm(x)
        x = x.view(B, num_regions, self.region_size, d_model)
        assert x.size(1) == num_regions and x.size(2) == self.region_size, "Reshaping in HATEncoder failed"
        if self.encoder_layout['pooling'] == 'None':
            memory = x.view(B, -1, d_model)
        else:
            memory = self.pooler(x)  # shape [B, num_regions, d_model]
        
        src_mask = torch.ones(B, 1, memory.size(1), dtype=torch.bool, device=x.device)
        assert src_mask.size(2) == memory.size(1), f"Mask length {src_mask.size(2)} != memory length {memory.size(1)}"
        return memory, src_mask


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    """
    Computes scaled dot-product attention in a memory‑efficient manner.
    Uses PyTorch’s built‑in scaled_dot_product_attention when available.
    Falls back to a chunked implementation if the full attention map is too large.
    """
    d_k = query.size(-1)
    # Use PyTorch's optimized function if available.
    if hasattr(F, "scaled_dot_product_attention"):
        attn_mask = None
        if mask is not None:
            # If mask shape is [B, 1, seq_k], expand along query length.
            if mask.dim() == 3 and mask.size(1) == 1:
                attn_mask = mask.expand(query.size(0), query.size(2), key.size(1))
            else:
                attn_mask = mask
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout.p if dropout is not None else 0.0,
            is_causal=False
        )
        return attn_output, None
    else:
        B, h, seq_q, _ = query.size()
        seq_k = key.size(-2)
        # Set a threshold for switching to chunked attention.
        threshold = 1024 * 1024  # adjust as needed
        if seq_q * seq_k > threshold:
            return chunked_attention(query, key, value, mask=mask, dropout=dropout, chunk_size=512)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            p_attn = F.softmax(scores, dim=-1)
            if dropout is not None:
                p_attn = dropout(p_attn)
            return torch.matmul(p_attn, value), p_attn

def chunked_attention(query, key, value, mask=None, dropout=None, chunk_size=512):
    """
    Splits the query into chunks to avoid allocating a huge attention map.
    """
    d_k = query.size(-1)
    B, h, seq_q, _ = query.size()
    outputs = []
    for i in range(0, seq_q, chunk_size):
        query_chunk = query[:, :, i:i+chunk_size, :]
        scores = torch.matmul(query_chunk, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # If mask shape is [B, 1, seq_k] then reuse; otherwise slice for the chunk.
            if mask.dim() == 3 and mask.size(1) != query_chunk.size(2):
                mask_chunk = mask
            else:
                mask_chunk = mask[:, i:i+chunk_size, :]
            scores = scores.masked_fill(mask_chunk == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        output_chunk = torch.matmul(p_attn, value)
        outputs.append(output_chunk)
    output = torch.cat(outputs, dim=2)
    return output, None

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MambaDecoderLayerWithCross(nn.Module):
    def __init__(self, d_model, dropout, d_state=None, d_conv=4, expand=4, num_heads=8):
        super().__init__()
        if d_state is None:
            d_state = d_model // 2
        # Self processing using a Mamba block (mimicking self-attention)
        self.self_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Cross attention branch: tokens attend to encoder memory.
        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        # Feed-forward block.
        self.ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        # Self branch: process x with Mamba.
        residual = x
        x = self.self_mamba(self.norm1(x))
        x = residual + self.dropout(x)
        
        # Cross attention: attend to encoder memory.
        residual = x
        x2 = self.cross_attn(self.norm2(x), memory, memory, mask=src_mask)
        x = residual + self.dropout(x2)
        
        # Feed-forward layer.
        residual = x
        x2 = self.ff(self.norm3(x))
        x = residual + self.dropout(x2)
        return x


class MambaDecoder(nn.Module):
    def __init__(self, d_model, num_layers, dropout, d_state=None, d_conv=4, expand=4, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaDecoderLayerWithCross(d_model, dropout, d_state, d_conv, expand, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Encoder_decoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, d_model, num_heads):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory, new_src_mask = self.encode(src, src_mask)
        return self.decode(memory, new_src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None):
        embeddings = self.tgt_embed(tgt)
        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.h = h
        # Create 4 linear layers for query, key, value, and final projection.
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None, layer_past=None):
        # Apply linear projections.
        query = self.linears[0](query)
        key   = self.linears[1](key)
        value = self.linears[2](value)
        
        nbatches_q = query.size(0)
        nbatches_k = key.size(0)
        if nbatches_k != nbatches_q:
            # If key (and value) have a smaller batch dimension than query,
            # repeat them so that their batch size matches query.
            key = key.repeat(nbatches_q, 1, 1)
            value = value.repeat(nbatches_q, 1, 1)
        
        nbatches = query.size(0)
        # Explicitly get the sequence lengths.
        L_query = query.size(1)
        L_key = key.size(1)
        L_value = value.size(1)
        
        # Reshape and transpose for multi-head attention.
        query = query.view(nbatches, L_query, self.h, self.d_k).transpose(1, 2)
        key   = key.view(nbatches, L_key, self.h, self.d_k).transpose(1, 2)
        value = value.view(nbatches, L_value, self.h, self.d_k).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # expand mask to cover all heads
        
        # Compute attention (using your custom attention function)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Concat heads and apply final linear layer.
        x = x.transpose(1, 2).contiguous().view(nbatches, L_query, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class CnvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.mamba = Mamba(d_model=latent_dim, d_state=d_state, d_conv=d_conv, expand=expand)
    def forward(self, x):
        # x can be [B, input_dim] or [B, seq_len, input_dim]
        x = self.input_proj(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.mamba(x)
        return x.mean(dim=1)

class BaseGen_SPECTRA(AttModel):
    def make_model(self, tgt_vocab, encoder_layout=None):
        c = copy.deepcopy
        # Use rotary positional encoding for target side.
        position = RotaryPositionalEncoding(self.d_model, self.dropout)
        model = Encoder_decoder(
            Encoder(encoder_layout),
            MambaDecoder(
                self.d_model, self.num_layers, self.dropout,
                d_state=self.d_model // 2, d_conv=4, expand=4, num_heads=self.num_heads
            ),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            self.d_model,
            self.num_heads
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(BaseGen_SPECTRA, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk
        self.K = args.prototype_num
        tgt_vocab = self.vocab_size + 1
        self.region_size = args.region_size
        self.beam_size = args.beam_size
        self.encoder_layout = {
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'region_size': self.region_size,
            'dropout': self.dropout,
            'pooling': 'attentive',
            'num_layers': 2,
            '0': {'region_encoder': True, 'WSI_encoder': True, 'first_layer': True},
            '1': {'region_encoder': True, 'WSI_encoder': False, 'first_layer': False},
        }
        self.model = self.make_model(tgt_vocab, self.encoder_layout)
        self.logit = nn.Linear(self.d_model, tgt_vocab)
        # CNV branch and cross-attention.
        self.cnv_encoder = CnvEncoder(input_dim=173, hidden_dim=self.d_model, latent_dim=self.d_model,
                                      d_state=16, d_conv=4, expand=2)
        self.cross_attn = MultiHeadedAttention(self.num_heads, self.d_model, dropout=self.dropout)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        return fc_feats, att_feats, att_feats, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        return att_feats, seq, att_masks, seq_mask


    def _forward(self, fc_feats, att_feats, cnv, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        memory, src_mask = self.model.encode(att_feats, att_masks)
        # Integrate CNV features via cross-attention.
        if cnv is not None:
            cnv_feature = self.cnv_encoder(cnv)            # [B, d_model]
            cnv_feature = cnv_feature.unsqueeze(1)         # [B, 1, d_model]
            # Expand cnv_feature to match the memory sequence length:
            cnv_feature = cnv_feature.expand(-1, memory.size(1), -1)  # [B, L, d_model]
            cross_feature = self.cross_attn(memory, cnv_feature, cnv_feature)
            memory = memory + cross_feature

        out = self.model.decode(memory, src_mask, seq, seq_mask)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def _save_attns(self, start=False):
        if start:
            self.attention_weights = []
        # For each layer, store the attention weights if they exist; otherwise, store None.
        self.attention_weights.append([
            layer.cross_attn.attn.cpu().numpy() if layer.cross_attn.attn is not None else None
            for layer in self.model.decoder.layers
        ])

    def _sample(self, fc_feats, att_feats, cnv, att_masks=None, update_opts={}):
        """
        If beam_size == 1 => run sampling (greedy/temperature).
        If beam_size > 1 => run beam search.
        """
        self.eval()
        with torch.no_grad():
            opt = self.args.__dict__
            opt.update(**update_opts)
            sample_method = opt.get('sample_method', 'greedy')
            beam_size = opt.get('beam_size', 1)
            temperature = opt.get('temperature', 1.0)
            sample_n = int(opt.get('sample_n', 1))
            group_size = opt.get('group_size', 1)
            output_logsoftmax = opt.get('output_logsoftmax', 1)
            decoding_constraint = opt.get('decoding_constraint', 0)
            block_trigrams = opt.get('block_trigrams', 0)
            cnv = cnv.to(fc_feats.device)
            att_feats_proc, seq_dummy, att_masks_proc, seq_mask = self._prepare_feature_forward(att_feats, att_masks, None)
            memory, src_mask = self.model.encode(att_feats_proc, att_masks_proc)

            if cnv is not None:
                cnv = cnv.to(fc_feats.device)
                cnv_feature = self.cnv_encoder(cnv)      # [B, d_model]
                cnv_feature = cnv_feature.unsqueeze(1)   # [B, 1, d_model]
                cnv_feature = cnv_feature.expand(-1, memory.size(1), -1)  # [B, L, d_model]
                cross_feature = self.cross_attn(memory, cnv_feature, cnv_feature)
                memory = memory + cross_feature

            batch_size = fc_feats.size(0)

            # ---------- Single-sample (beam_size==1) branch ----------
            if beam_size == 1:
                state = self.init_hidden(batch_size * sample_n)
                if sample_n > 1:
                    memory, src_mask = utils.repeat_tensors(sample_n, [memory, src_mask])
                seq = fc_feats.new_full((batch_size * sample_n, self.max_seq_length), self.pad_idx, dtype=torch.long)
                seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.max_seq_length, self.vocab_size + 1)
                unfinished = fc_feats.new_ones(batch_size * sample_n, dtype=torch.bool)
                for t in range(self.max_seq_length + 1):
                    if t == 0:
                        it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)
                    logprobs, state = self.get_logprobs_state_with_memory(it, memory, src_mask, state,
                                                                        output_logsoftmax=output_logsoftmax)
                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:, t - 1].unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp
                    if block_trigrams and t >= 3:
                        pass
                    if t == self.max_seq_length:
                        break
                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)
                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        it[~unfinished] = self.pad_idx
                        logprobs = logprobs * unfinished.unsqueeze(1).float()
                        unfinished = unfinished & (it != self.eos_idx)
                    seq[:, t] = it
                    seqLogprobs[:, t] = logprobs
                    if unfinished.sum() == 0:
                        break
                return seq, seqLogprobs

            # ---------- Beam search branch (beam_size > 1) ----------
            else:
                seqs = []
                seqLogprobs_all = []
                for b in range(batch_size):
                    mem_b = memory[b:b+1]       # [1, L, d_model]
                    src_mask_b = src_mask[b:b+1]  # [1, 1, L] or similar
                    mem_b = mem_b.expand(beam_size, mem_b.size(1), mem_b.size(2)).contiguous()
                    src_mask_b = src_mask_b.expand(beam_size, src_mask_b.size(1), src_mask_b.size(2)).contiguous()

                    beam_seq = fc_feats.new_full((beam_size, self.max_seq_length), self.pad_idx, dtype=torch.long)
                    beam_seq_logprobs = fc_feats.new_zeros(beam_size, self.max_seq_length)
                    beam_logprobs_sum = fc_feats.new_zeros(beam_size)
                    beam_state = [self.init_hidden(beam_size)]
                    it = fc_feats.new_full((beam_size,), self.bos_idx, dtype=torch.long)
                    done_beams = []

                    for t in range(self.max_seq_length):
                        if t == 0:
                            logprobs, new_state = self.get_logprobs_state_with_memory(
                                it, mem_b, src_mask_b, beam_state[-1], output_logsoftmax=output_logsoftmax
                            )
                            # At t==0, we only have one real path; pick top tokens from logprobs[0]
                            topk_logprobs, topk_ids = logprobs[0].topk(beam_size, dim=0)
                            for k in range(beam_size):
                                beam_seq[k, t] = topk_ids[k]
                                beam_seq_logprobs[k, t] = topk_logprobs[k]
                                beam_logprobs_sum[k] = topk_logprobs[k]
                            new_state_list = []
                            for st_item in new_state:
                                if isinstance(st_item, torch.Tensor):
                                    # Here st_item shape is [1, beam_size, ...]; we leave dimension 0 intact.
                                    # For a 4-D tensor: [1, beam_size, seq_len, d_model]
                                    # For a 3-D tensor: [1, beam_size, d_model]
                                    new_state_list.append(st_item.clone())
                                elif isinstance(st_item, list):
                                    expanded_list = []
                                    for s in st_item:
                                        expanded_list.append(s.clone())
                                    new_state_list.append(expanded_list)
                                else:
                                    new_state_list.append(st_item)
                            beam_state.append(new_state_list)
                        else:
                            logprobs, new_state = self.get_logprobs_state_with_memory(
                                beam_seq[:, t-1], mem_b, src_mask_b, beam_state[-1],
                                output_logsoftmax=output_logsoftmax
                            )
                            cand_logprobs = logprobs + beam_logprobs_sum.unsqueeze(1)
                            flat_cand_logprobs = cand_logprobs.view(-1)
                            topk_logprobs, topk_ids = flat_cand_logprobs.topk(beam_size, dim=0)
                            new_beam_seq = beam_seq.clone()
                            new_beam_seq_logprobs = beam_seq_logprobs.clone()
                            new_beam_logprobs_sum = beam_logprobs_sum.clone()
                            new_beam_state_list = []
                            for st_item in new_state:
                                if isinstance(st_item, torch.Tensor):
                                    new_beam_state_list.append(st_item.clone())
                                elif isinstance(st_item, list):
                                    new_list = []
                                    for s in st_item:
                                        new_list.append(s.clone())
                                    new_beam_state_list.append(new_list)
                                else:
                                    new_beam_state_list.append(st_item)
                            vocab_size = logprobs.size(1)
                            for i, (val, idx) in enumerate(zip(topk_logprobs, topk_ids)):
                                beam_idx = idx // vocab_size
                                token_idx = idx % vocab_size

                                new_beam_seq[i, :t] = beam_seq[beam_idx, :t]
                                new_beam_seq[i, t] = token_idx
                                new_beam_seq_logprobs[i, :t] = beam_seq_logprobs[beam_idx, :t]
                                new_beam_seq_logprobs[i, t] = logprobs[beam_idx, token_idx]
                                new_beam_logprobs_sum[i] = val

                                for j, st_item in enumerate(new_state):
                                    if isinstance(st_item, torch.Tensor):
                                        # st_item has shape [1, beam_size, ...]; update beam dimension at index i
                                        new_beam_state_list[j][0, i] = st_item[0, beam_idx]
                                    elif isinstance(st_item, list):
                                        for layer_ix in range(len(st_item)):
                                            new_beam_state_list[j][layer_ix][0, i] = st_item[layer_ix][0, beam_idx]
                            beam_seq = new_beam_seq
                            beam_seq_logprobs = new_beam_seq_logprobs
                            beam_logprobs_sum = new_beam_logprobs_sum
                            beam_state[-1] = new_beam_state_list

                        eos_mask = (beam_seq[:, t] == self.eos_idx)
                        if t == self.max_seq_length - 1:
                            eos_mask.fill_(1)
                        for k in range(beam_size):
                            if eos_mask[k]:
                                done_beams.append({
                                    'seq': beam_seq[k].clone(),
                                    'logprob': beam_logprobs_sum[k].item()
                                })
                                beam_logprobs_sum[k] = -9999
                        if eos_mask.all():
                            break
                    if len(done_beams) == 0:
                        for k in range(beam_size):
                            done_beams.append({
                                'seq': beam_seq[k].clone(),
                                'logprob': beam_logprobs_sum[k].item()
                            })
                    done_beams = sorted(done_beams, key=lambda x: -x['logprob'])
                    best_beam = done_beams[0]
                    seqs.append(best_beam['seq'].unsqueeze(0))
                    seq_logprobs_best = fc_feats.new_zeros(self.max_seq_length, dtype=torch.float)
                    seqLogprobs_all.append(seq_logprobs_best.unsqueeze(0))
                seq = torch.cat(seqs, dim=0)
                seqLogprobs = torch.cat(seqLogprobs_all, dim=0)
                return seq, seqLogprobs

    def get_logprobs_state_with_memory(self, it, memory, src_mask, state, output_logsoftmax=1):
        xt = self.embed(it)
        output, state = self.core_with_memory(xt, memory, src_mask, state)
        if len(output.shape) > 2:
            output = output.squeeze(1)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)
        return logprobs, state

    def core_with_memory(self, xt, memory, src_mask, state):
        """
        The 'core' step of decoding with the provided memory/src_mask.
        """
        if len(state) == 0:
            ys = xt.unsqueeze(1)
            past = [
                xt.new_zeros(self.num_layers * 2, xt.shape[0], 0, self.d_model),
                xt.new_zeros(self.num_layers * 2, xt.shape[0], 0, self.d_model)
            ]
        else:
            ys = torch.cat([state[0][0], xt.unsqueeze(1)], dim=1)
            past = state[1:]
        out = self.model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).to(memory.device), past=past)
        if not self.training:
            self._save_attns(start=(len(state) == 0))
        return out[:, -1], [ys.unsqueeze(0)] + past
