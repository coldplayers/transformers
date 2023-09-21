import torch
import torch.nn as nn


# evaluation function
def local_heavy_hitter_recent_mask(attn_weights, heavy_budget, recent_budget, min_val, no_padding_seq_length=None, return_scores_masks=False):

    # attn_weights (head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    if seq_length <= heavy_budget + recent_budget:
        recent_budget = int(seq_length/2)
        heavy_budget = seq_length - recent_budget
    if no_padding_seq_length is None:
        padding_length = 0
    else:
        padding_length = seq_length - no_padding_seq_length

    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    accumulated_attention_score = torch.sum(tmp_attn[:,padding_length:heavy_budget+recent_budget+padding_length,:], dim=-2) #(head, keys)
    accumulated_attention_score[:,heavy_budget+recent_budget+padding_length:] = 0

    if padding_length > 0:
        accumulated_attention_score[:,:padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[:, padding_length:heavy_budget+recent_budget+padding_length, padding_length:heavy_budget+recent_budget+padding_length] = True
    
    k = recent_budget + heavy_budget - 1
    position_relative = False
    for token_index in range(heavy_budget+recent_budget-1+padding_length, seq_length):
        # import pdb;pdb.set_trace()

        # tmp_attn_index = nn.functional.softmax(attn_weights[:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        tmp_attn_index = tmp_attn[:,token_index,:]
        inf_score = torch.zeros_like(accumulated_attention_score)
        inf_score[:,max(token_index-recent_budget+1,0):token_index] = 1e10
        # if token_index < recent_budget + heavy_budget:
        #     k = token_index
        # else:
        #     k = recent_budget + heavy_budget - 1
        if position_relative:
            alpha = torch.arange(1, token_index+2).flip(dims=[0]).to(tmp_attn.device)
            alpha = torch.cat([1/alpha, torch.zeros(seq_length - token_index - 1, dtype=alpha.dtype, device=alpha.device)])
            alpha = alpha.unsqueeze(0)
        else:
            alpha = 1
        _, tmp_topk_index = (accumulated_attention_score*alpha+inf_score).topk(k=k, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        mask_bottom_index[:, token_index] = True

        mask_bottom[:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index
    
    # select topk in accumulated_attention_score and set the rest to 0
    attention_masks_next = torch.zeros(accumulated_attention_score.shape[0], accumulated_attention_score.shape[1]+1).to(dtype_attn_weights).to(attn_weights.device)
    index = accumulated_attention_score[:,:-recent_budget+1].topk(k=heavy_budget, dim=-1)[1]
    attention_masks_next = attention_masks_next.scatter(-1, index, 1)
    attention_masks_next[:,:-recent_budget] = 1
    attention_masks_next = attention_masks_next.unsqueeze(1)

    mask_bottom = torch.tril(mask_bottom, diagonal=0)
    attn_weights[~mask_bottom] = min_val
    outputs = (attn_weights,)
    if return_scores_masks:
        outputs = outputs + (accumulated_attention_score, attention_masks_next)
    return outputs


# # training function
# def local_heavy_hitter_recent_mask(attn_weights, heavy_budget, recent_budget, min_val, no_padding_seq_length=None):
#     # attn_weights (head, query, keys)
#     dtype_attn_weights = attn_weights.dtype
#     seq_length = attn_weights.shape[-1]
#     if no_padding_seq_length is None:
#         padding_length = 0
#     else:
#         padding_length = seq_length - no_padding_seq_length

#     tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)
#     device = tmp_attn.device

#     accumulated_attention_score = torch.sum(tmp_attn[:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
#     accumulated_attention_score[:,heavy_budget+padding_length:] = 0

#     if padding_length > 0:
#         accumulated_attention_score[:,:padding_length] = 0

#     mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
#     mask_bottom[:, padding_length:heavy_budget+padding_length, padding_length:heavy_budget+padding_length] = True
#     mask_bottom_slice = mask_bottom[:,heavy_budget+padding_length:,:]
#     all_inf = torch.ones(mask_bottom.shape, device=device)
#     inf_score_mask = torch.logical_and(torch.triu(all_inf, diagonal = -recent_budget), torch.tril(all_inf,diagonal = 0))
#     inf_score = torch.zeros(inf_score_mask.shape, device=device)
#     inf_score[inf_score_mask] = float('inf')
#     inf_score = inf_score[:,heavy_budget+padding_length:,:]
#     accumulated_attention_score_matrix = torch.cumsum(tmp_attn[:,heavy_budget+padding_length:,:],dim = -2) + accumulated_attention_score.unsqueeze(-2)
#     _, topk_index = (accumulated_attention_score_matrix+inf_score).topk(k=heavy_budget+recent_budget, dim=-1)
#     zeros_index = torch.zeros_like(mask_bottom_slice, dtype = torch.bool)
#     mask_bottom_slice_index = zeros_index.scatter(-1,topk_index, True)
#     mask_bottom = torch.cat([mask_bottom[:,:heavy_budget+padding_length],mask_bottom_slice_index],dim = 1)
#     mask_bottom = torch.tril(mask_bottom, diagonal=0)
#     attn_weights[~mask_bottom] = min_val
#     return attn_weights


def ar_heavy_hitter_recent_mask(attn_weights, previous_scores, heavy_budget, recent_budget, no_padding_seq_length=None):
    # attn_weights (heads, q-tokens, k-tokens) 16, 15, 15 // 16, 1, 16
    current_scores_sum = attn_weights.sum(1) # (heads, k-tokens)

    cache_budget = heavy_budget + recent_budget
    # Accumulate attention scores
    if not previous_scores == None:
        current_scores_sum[:, :-1] += previous_scores #(Enlarge Sequence)
    dtype_attn_weights = attn_weights.dtype
    attn_weights_devices = attn_weights.device

    previous_scores = current_scores_sum #(heads, k-tokens)
    attn_mask = torch.ones(current_scores_sum.shape[0], current_scores_sum.shape[1]+1).to(dtype_attn_weights).to(attn_weights_devices)

    attn_tokens_all = previous_scores.shape[-1]
    if attn_tokens_all >= cache_budget:
        # activate most recent k-cache
        inf_score = torch.zeros_like(previous_scores)
        inf_score[:,-recent_budget+1:] = float('inf')
        _, keep_topk = (previous_scores+inf_score).topk(k=heavy_budget+recent_budget-1, dim=-1, largest=True)
        attn_mask = attn_mask.scatter(-1, keep_topk, 0)
        # ~attn_mask
        attn_mask[:,:-1] = 1-attn_mask[:,:-1]

    attention_masks_next = attn_mask.unsqueeze(1)

    score_mask = attn_mask[:,:-1]
    previous_scores = previous_scores * score_mask
    return attention_masks_next, previous_scores
