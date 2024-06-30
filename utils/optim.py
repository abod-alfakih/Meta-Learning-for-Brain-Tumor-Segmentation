import torch
import torch.nn as nn

def get_optimizer(args, net: nn.Module):
    lr, weight_decay = args.lr, args.weight_decay

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, nesterov=False,dampening=0.)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)     # default is adam

    return optimizer

def get_clean_optimizer(args, net: nn.Module):
    lr, weight_decay = args.lr, args.weight_decay

    if args.clean_optim == 'adamw':
        clean_optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=weight_decay)
    elif args.clean_optim == 'sgd':
        clean_optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, nesterov=True)
    else:
        clean_optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)     # default is adam

    return clean_optimizer
