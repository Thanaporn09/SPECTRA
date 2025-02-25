import torch
from transformers import get_linear_schedule_with_warmup

def build_optimizer(args, model):

    if args.model == "BLIP":
        ve_params = list(map(id, model.encoder_decoder.model.encoder.parameters()))
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
        optimizer = getattr(torch.optim, args.optim)(
             [{'params': model.encoder_decoder.model.encoder.parameters(), 'lr': args.lr_ve},
              {'params': ed_params, 'lr': args.lr_ed}],
              weight_decay=args.weight_decay,
              amsgrad=args.amsgrad
            )
    elif args.model == "BLIP2":
        ve_params = list(map(id, model.encoder_decoder.model.encoder.parameters()))
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
        optimizer = getattr(torch.optim, args.optim)(
             [{'params': model.encoder_decoder.model.encoder.parameters(), 'lr': args.lr_ve},
              {'params': ed_params, 'lr': args.lr_ed}],
              weight_decay=args.weight_decay,
              amsgrad=args.amsgrad
            )
    else:
        ed_params = model.parameters()#filter(lambda x: id(x) not in ve_params, model.parameters())
        optimizer = getattr(torch.optim, args.optim)(
            #[{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
            [{'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    return optimizer


def build_lr_scheduler(args, optimizer):
    # if args.model == 'BLIP2':
    #     warmup_steps = int(0.1 * args.epochs)
    #     lr_scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=warmup_steps,
    #         num_training_steps=args.epochs
    #     )
    # else:
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
