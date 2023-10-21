# coding=utf-8
import os
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

logger = logging.getLogger(__name__)


def build_optimizer_and_scheduler(args, model, t_total, optim="adamw", schd="lstep"):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        elif space[0] == 'crf':
            crf_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_lr},
    ]

    if optim.lower() == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    elif optim.lower() == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=args.adam_epsilon,
              )
    
    if schd == "step":
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.999)
    elif schd == "lstep":
      scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
      )
    else:
      scheduler = None
    
    return optimizer, scheduler

def save_model(args, model, model_name, global_step):
    """保存最好的验证集效果最好那个模型"""
    output_dir = os.path.join(args.output_dir, '{}'.format(model_name, global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info('Saving model checkpoint to {}'.format(output_dir))
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

def save_model_step(args, model, global_step):
    """根据global_step来保存模型"""
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info('Saving model & optimizer & scheduler checkpoint to {}.format(output_dir)')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """
    gpu_ids = gpu_ids.split(',')

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if ckpt_path is not None:
        logger.info('Load ckpt from {}'.format(ckpt_path))
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    model.to(device)

    if len(gpu_ids) > 1:
        logger.info('Use multi gpus in: {}'.format(gpu_ids))
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info('Use single gpu in: {}'.format(gpu_ids))

    return model, device