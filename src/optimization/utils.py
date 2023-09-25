from torch.optim import Adam, Adamax, SGD
from src.optimization.adamw import AdamW


def setup_e2e_optimizer(model, opts):
    optimizer_grouped_parameters = build_e2e_optimizer_w_lr_mul(
        model = model,
        learning_rate = opts.learning_rate,
        weight_decay = opts.weight_decay,
        lr_mul = opts.lr_mul
    )
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters, lr=opts.learning_rate, betas=opts.betas)

    param_group_field = "lr"
    for i, group in enumerate(optimizer.param_groups):
        if param_group_field not in group:
            raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
        group.setdefault(f"initial_{param_group_field}", group[param_group_field])

    return optimizer


def build_e2e_optimizer_w_lr_mul(
        model, learning_rate, weight_decay,
        lr_mul=1, lr_mul_prefix=""):
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       ((not any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": weight_decay,
            "lr": learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       ((any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": 0.0,
            "lr": learning_rate
        },
        {
            "params": [p for n, p in model.visual_encoder.named_parameters() if
                       ((not any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": weight_decay,
            "lr": lr_mul * learning_rate
        },
        {
            "params": [p for n, p in model.visual_encoder.named_parameters() if
                       ((any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": 0.0,
            "lr": lr_mul * learning_rate
        },

    ]
    return optimizer_grouped_parameters