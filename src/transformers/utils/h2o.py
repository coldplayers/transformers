import torch
import torch.nn as nn


def local_heavy_hitter_recent_mask(attn_weights, heavy_budget, recent_budget, min_val, no_padding_seq_length=None):

    # attn_weights (head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    if no_padding_seq_length is None:
        padding_length = 0
    else:
        padding_length = seq_length - no_padding_seq_length

    offset = torch.finfo(attn_weights.dtype).min
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    accumulated_attention_score = torch.sum(tmp_attn[:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
    accumulated_attention_score[:,heavy_budget+padding_length:] = 0

    if padding_length > 0:
        accumulated_attention_score[:,:padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[:, padding_length:heavy_budget+padding_length, padding_length:heavy_budget+padding_length] = True

    for token_index in range(heavy_budget+padding_length, seq_length):

        tmp_attn_index = nn.functional.softmax(attn_weights[:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        inf_score = torch.zeros_like(accumulated_attention_score)
        inf_score[:,max(token_index-recent_budget,0):token_index] = 1e10
        _, tmp_topk_index = (accumulated_attention_score+inf_score).topk(k=heavy_budget+recent_budget-1, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        mask_bottom_index[:, token_index] = True

        mask_bottom[:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index

    mask_bottom = torch.tril(mask_bottom, diagonal=0)
    attn_weights[~mask_bottom] = min_val
    return attn_weights