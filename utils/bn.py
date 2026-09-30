import torch
import torch.nn as nn


def inject_bn(model, bn_layer: nn.BatchNorm2d):
    for model_key in model.keys():
        # /* dynamic BN
        BN_to_inject = bn_layer
        
        n_bn = count_bn(model[model_key], torch.nn.BatchNorm2d)
        n_bn_to_replace = int(n_bn * 1.0)

        n_repalced = replace_bn(model[model_key], BN_to_inject,
                                number_to_replace=n_bn_to_replace,
                                beta=0.1,
                                bn_dist_scale= 0.1, #0.1,
                                smoothing_beta=0.2,
                    )
        assert n_repalced == n_bn_to_replace, f"Replaced {n_repalced} BNs but you wanted to replace {n_bn_to_replace}. Need to update `replace_bn`."

        n_bn_inside = count_bn(model[model_key], BN_to_inject)
        assert n_repalced == n_bn_inside, f"Replaced {n_repalced} BNs but actually inserted {n_bn_inside} {BN_to_inject.__name__}."
        model[model_key].cuda()


def replace_bn(model: nn.Module, BN_module: nn.Module, n_repalced=0, number_to_replace=None, **abn_kwargs):
    copy_keys = ['eps', 'momentum', 'affine']

    for mod_name, target_mod in model.named_children():
        if number_to_replace is not None and n_repalced == number_to_replace:
            return n_repalced
        
        if isinstance(target_mod, nn.BatchNorm2d) or isinstance(target_mod, nn.SyncBatchNorm):
            n_repalced += 1
            
            new_mod = BN_module(
                target_mod.num_features,
                **{k: getattr(target_mod, k) for k in copy_keys},
                **abn_kwargs,
            )
            new_mod.load_state_dict(target_mod.state_dict())
            new_mod.track_running_stats = False
            setattr(model, mod_name, new_mod)
        else:
            n_repalced = replace_bn(
                target_mod, BN_module, n_repalced=n_repalced, number_to_replace=number_to_replace, **abn_kwargs)
    return n_repalced

def count_bn(model: nn.Module, BN_module):
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, BN_module):
            cnt += 1
    return cnt