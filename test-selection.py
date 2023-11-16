def select_test_views(local_feat, clip_weights):
    norm_local_feat = local_feat / local_feat.norm(dim=-1, keepdim=True)
    local_logits = norm_local_feat @ clip_weights
    logits_values, _ = torch.topk(local_logits, k=2, dim=-1)
    criterion = logits_values[:,:,0] - logits_values[:,:,1]
    local_idx = torch.argsort(criterion, dim=0, descending=True)[:1]
    selected = torch.take_along_dim(local_feat, local_idx[:,:,None], dim=0).squeeze(0)
    return selected
