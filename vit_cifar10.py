import torch
import torch.nn as nn

import time

from data import get_dataloaders
from vit import VisionTransformer
import torch.optim as optim
from tqdm import tqdm

from utils import *

import os
import parser
import random
import warnings
import wandb


def train(opt, train_loader, model, criterion, optimizer):
    """One epoch training"""
    model.train()

    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    losses = AverageMeter('Loss')

    end = time.time()

    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for idx, (inputs, targets) in enumerate(pbar):

            inputs = inputs.float()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            # ===================forward=====================

            logits = model(inputs)
            loss = criterion(logits, targets)

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # ===================backward=====================

            # if opt.amp:
            #     step_ = idx + epoch*len(train_loader)
            #     scaler.scale(loss).backward()
            #     if (step_ + 1) % 1 == 0:
            #         scaler.step(optimizer)
            #         scaler.update()
            #         optimizer.zero_grad()
            # else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================logging=====================

            pbar.set_postfix({"Acc@1": '{0:.2f}'.format(top1.avg.cpu().numpy()),
                              "Acc@5": '{0:.2f}'.format(top5.avg.cpu().numpy(), 2),
                              "Loss": '{0:.2f}'.format(losses.avg, 2),
                              })

    return top1.avg, top5.avg, losses.avg


def test(opt, model, test_loader, criterion):
    """One epoch testing"""
    model.eval()

    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    losses = AverageMeter('Loss')

    start = time.time()

    with torch.zero_grad():
        with tqdm(test_loader, total=len(test_loader)) as pbar:
            for idx, (inputs, targets) in enumerate(pbar):

                inputs = inputs.float()

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                # ===================forward=====================

                logits = model(inputs)
                loss = criterion(logits, targets)

                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

                # ===================logging=====================

                pbar.set_postfix({"Acc@1": '{0:.2f}'.format(top1.avg.cpu().numpy()),
                                  "Acc@5": '{0:.2f}'.format(top5.avg.cpu().numpy(), 2),
                                  "Loss": '{0:.2f}'.format(losses.avg, 2),
                                  })

        return top1.avg, top5.avg, losses.avg


def main(args):

    # args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = args.wandb_key
    os.environ['WANDB_MODE'] = args.mode

    if args.resume:
        wandb.init(project="capsnet", id=args.id, resume="true")
    else:
        args.id = wandb.util.generate_id()
        wandb.init(project="capsnet", id=args.id, resume="allow")
        wandb.config.update(args)
        try:
            path = os.path.join(wandb.run.dir, "codes")
            os.system('mkdir ' + path)
            os.syesem('cp -r *.py '+path)
            wandb.save("*.py")
        except:
            pass

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')

    train_loader, test_loader, classes = get_dataloaders()

    cfg = {
        "img_size": 32,
        "in_chans": 3,
        "patch_size": 4,
        "embed_dim": 512,
        "n_classes": 10,
        "depth": 6,
        "n_heads": 8,
        "qkv_bias": True,
        "mlp_ratio": 1,
        "p": 0.1,
        "attn_p": 0.1,
    }

    model = VisionTransformer(**cfg)

    print(f'Total parameter: {0:.2f}M'.format(sum(
        p.numel()/1000000 for p in model.parameters())))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    num_epochs = 100

    for epoch in range(num_epochs):

        print("\nEpoch => {} ".format(epoch), '='*80, "\n")

        # train
        start = time.time()
        train_acc, train_acc5, train_loss = train(
            "", train_loader, model, criterion, optimizer)
        train_time = time.time() - start

        # test
        start = time.time()
        test_acc, test_acc5, test_loss = test(
            "", test_loader, model, criterion)
        test_time = time.time() - start

        # save
        is_best = test_acc > best_acc1
        best_acc1 = max(test_acc, best_acc1)

        model_name = wandb.run.id+"_vit.pt"

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best,)

        wandb.log({'epoch': epoch,
                   'Train Acc': train_acc,
                   'Train Acc @5': train_acc5,
                   'Train Loss': train_loss,
                   'Test Acc': test_acc,
                   'Test Acc @5': test_acc5,
                   'Test Loss': test_loss,
                   #    'Learning Rate': scheduler.get_last_lr()[0],
                   'Training Time': train_time,
                   'Testing Time': test_time
                   })


if __name__ == "__main__":

    class Args:
        def __init__(self):

            self.wandb_key = "5a31e6e19541c40a040a39b6899e7c6ee307082e"
            self.mode = "offline"
            self.resume = False

    args = Args()

    main(args)
