import builtins
from datetime import datetime
import wandb

import os
import sys
import random
import shutil
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from nets.utils import get_model, load_pretrain, load_resume, ExponentialMovingAverage, auto_resume_helper
from datasets.utils import get_train_dataset, get_val_dataset
from datasets.imbalanced_sampler import BalanceClassSampler, DistributedSamplerWrapper
from torch.utils.data import DataLoader
from datasets.DataLoaderX import DataLoaderX
from utils import reduce_tensor, AverageMeter, ProgressMeter, concat_all_gather
from logger import create_logger
import logging
from losses.single_center_loss import SingleCenterLoss
import numpy as np

from config import config as cfg


best_acc1 = 0

def init_logger_wandb(rank):
    wandb_logger = None

    run_name = datetime.now().strftime("%y%m%d_%H") + f"_GPU{rank}_" + cfg.suffix_run_name
    
    run_exists = False
    run_id = None
    wandb.login(key=cfg.wandb_key)
    try:
        api = wandb.Api()
        runs = api.runs(path=f"{cfg.wandb_entity}/{cfg.wandb_project}")
        for run in runs:
            if run.name == run_name:
                run_exists = True
                run_id = run.id  # Get the run ID to resume
                break
    except Exception as e:
        print("WandB API error while checking existing runs.")
        print(f"API Error: {e}")

    # Resume if run_name exists
    try:
        wandb_logger = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            sync_tensorboard=True,
            resume="must" if run_exists else False,  # Ensures resume if run exists
            id=run_id if run_exists else None,  # Use the existing run ID if found
            name=run_name,
            notes=cfg.notes
        )
        print(f"wandb_logger init success ::: resume ::: {run_exists}")
    except Exception as e:
        print("WandB Data (Entity and Project name) must be provided in config file (config.py).")
        print(f"Wandb Error: {e}")

    return wandb_logger

def main():
    global best_acc1

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if rank == 0:
        if not os.path.exists(cfg.saved_model_dir):
            os.makedirs(cfg.saved_model_dir)

    time.sleep(3) # ensure dir create success
    create_logger(output_dir=cfg.saved_model_dir, dist_rank=dist.get_rank())

    if int(os.environ["LOCAL_RANK"]) != 0:
        def print_pass(*cfg):
            pass
        builtins.print = print_pass
    logging.info('params: {}'.format(cfg))
    logging.info('world_size:{}, rank:{}, local_rank:{}'.format(world_size, rank, int(os.environ["LOCAL_RANK"])))

    # 设置一个随机数种子
    seed_value = 240
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info('seed_value: {}'.format(seed_value))
    def worker_init_fn(worker_id):
        np.random.seed(seed_value + worker_id)


    # train dataset
    train_dataset = get_train_dataset(cfg.root, os.path.join(cfg.root, cfg.train_list), cfg.input_size)
    logging.info('train_dataset:{}'.format(len(train_dataset)))

    if cfg.imbalanced_sampler:
        logging.info('with imbalanced sampler')
        train_sampler = DistributedSamplerWrapper(BalanceClassSampler(train_dataset, mode='upsampling'))
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoaderX(
        local_rank=int(os.environ["LOCAL_RANK"]), dataset=train_dataset, batch_size=cfg.batch_size//world_size,
        shuffle=False, num_workers=cfg.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)
    logging.info('train_loader:{}'.format(len(train_loader)))

    # validate dataset
    val_dataset = get_val_dataset(cfg.root, os.path.join(cfg.root, cfg.val_list), cfg.input_size)
    logging.info('val_dataset: {}'.format(len(val_dataset)))

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoaderX(
        local_rank=int(os.environ["LOCAL_RANK"]), dataset=val_dataset, batch_size=cfg.batch_size//world_size,
        shuffle=False, num_workers=cfg.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    logging.info('val_loader:{}'.format(len(val_loader)))

    cfg.warmup_steps = len(train_loader) * cfg.warmup_epochs
    cfg.total_steps = len(train_loader) * cfg.epochs
    logging.info(f'warmup_steps: {cfg.warmup_steps}')
    logging.info(f'total_steps: {cfg.total_steps}')

    # create model
    model = None
    logging.info("=> creating model '{}', fp16:{}".format(cfg.arch, cfg.fp16))
    model = get_model(cfg.arch, cfg.num_classes, cfg.fp16)

    # if cfg.pretrain:
    #     load_pretrain(cfg.pretrain, model)

    model.cuda()
    if cfg.syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('use syncbn')

    # freeze backbond
    # i = 0
    # for param in model.parameters():
    #     i += 1
    #     if i < 161:
    #         param.requires_grad = False
    # print("Backbone frozen::: ", i)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    criterions = {}



    weights = torch.tensor([cfg.live_weight, 1.0])
    criterions['criterion_ce'] = nn.CrossEntropyLoss(weight=weights).cuda(int(os.environ["LOCAL_RANK"]))

    if cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                    eps=1.0e-08, betas=[0.9, 0.999],
                                    lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    eps=1.0e-08, betas=[0.9, 0.999],
                                    lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise 'unkown optimizer'

    scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=400)

    if cfg.resume:
        load_resume(cfg, model, optimizer, scaler)
        # validate(val_loader, model, criterions)
        # print("="*50)
        # print("="*50)
        # print("="*50)
        # print("Validation first")
        # print("="*50)
        # print("="*50)
        # print("="*50)

    # cfg.resume = "./outputs/ZZZ_1109_LIVE_WEIGHT_1.0_addblur_resnet/resnet50_epoch000_acc1_98.1237.pth"
    # best_acc1 = load_resume(cfg, model, optimizer, scaler)
    # validate(val_loader, model, criterions)
    # print("="*50)train
    # print("="*50)
    # print("="*50)
    # print("Validation first")
    # print("="*50)
    # print("="*50)
    # print("="*50)

    # Init log wandb
    wandb_logger = None
    if rank == 0:
        wandb_logger = init_logger_wandb(rank)

    start_epoch = 0
    for epoch in range(start_epoch, cfg.epochs):
        train_sampler.set_epoch(epoch)

        train_loss, train_acc1 = train(train_loader, model, criterions, optimizer, epoch, scaler, wandb_logger)

        if (epoch + 1) % cfg.save_freq == 0 or epoch + 1 == cfg.epochs:
            val_loss, val_acc1 = validate(val_loader, model, criterions)
            if rank == 0:
                state = {
                    'epoch': epoch + 1,
                    'arch': cfg.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scaler' : scaler.state_dict()
                }
                if not os.path.exists(cfg.saved_model_dir):
                    os.makedirs(cfg.saved_model_dir)
                save_path = os.path.join(cfg.saved_model_dir, f'{cfg.arch}_epoch{epoch:0>3d}_acc1_{val_acc1:.4f}.pth')
                torch.save(state, save_path)

                if val_acc1 > best_acc1:
                    best_acc1 = val_acc1
                    shutil.copyfile(save_path, os.path.join(cfg.saved_model_dir, f'{cfg.arch}_bestacc1_{best_acc1:.4f}_epoch{epoch:0>3d}.pth'))

                if wandb_logger is not None:
                    wandb_logger.log({
                    'Val/Loss': val_loss,
                    'Val/Acc@1': val_acc1,
                    })

                    if cfg.save_artifacts == True:
                        # Log the model checkpoint as an artifact
                        artifact = wandb.Artifact(name=f'{cfg.arch}_epoch{epoch:0>3d}', type='model')
                        artifact.add_file(save_path)
                        wandb_logger.log_artifact(artifact)

                        # Optionally, log the best model as well
                        if val_acc1 > best_acc1:
                            best_artifact = wandb.Artifact(name=f'{cfg.arch}_best_model', type='model')
                            best_artifact.add_file(best_model_path)
                            wandb_logger.log_artifact(best_artifact)

                #writer.add_scalar('Train_Loss/epochs', train_loss, epoch)
                #writer.add_scalar('Train_acc1/epochs', train_acc1, epoch)
                #writer.add_scalar('Val_Loss/epochs', val_loss, epoch)
                #writer.add_scalar('Val_acc1/epochs', val_acc1, epoch)
                #writer.add_scalar('lr/epochs', lr, epoch)
    dist.destroy_process_group()


def train(train_loader, model, criterions, optimizer, epoch, scaler, wandb_logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ce = AverageMeter('Loss_ce', ':.5f')
    losses_scl = AverageMeter('Loss_scl', ':.5f')

    losses = AverageMeter('Loss', ':.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, top1, losses_ce, losses_scl, losses] if cfg.single_center_loss 
            else [batch_time, data_time, top1, losses_ce, losses],)

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        global_step = epoch * len(train_loader) + i
        lr = adjust_learning_rate(optimizer, global_step, epoch)

        images = images.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)
        target = target.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)

        feats, output = model(images)
        loss_ce = criterions['criterion_ce'](output, target)
        if cfg.single_center_loss:
            if 'criterion_scl' not in criterions:
                D = feats.shape[1]
                # logger.info(f'Create single center loss, features dim: {D}')
                criterions['criterion_scl'] = SingleCenterLoss(m=0.3, D=D, use_gpu=True)
            loss_scl = criterions['criterion_scl'](feats, target)
            loss = loss_ce + cfg.single_center_loss_weight * loss_scl

            losses_scl.update(loss_scl.item(), images.size(0))
        else:
            loss = loss_ce

        # output = model(images)
        # loss_ce = criterions['criterion_ce'](output, target)
        # loss = loss_ce

        acc1, acc5 = accuracy(output, target, topk=(1, min(5, cfg.num_classes)))
        top1.update(acc1[0], images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses.update(loss.item(), images.size(0))


        loss /= cfg.accumulate_step
        if cfg.fp16:
            scaler.scale(loss).backward()
            if (i + 1) % cfg.accumulate_step == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (i + 1) % cfg.accumulate_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
        if (i + 1) % cfg.accumulate_step == 0:
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            logging.info(f'epoch {epoch} lr {lr:.6f} ' + progress.display(i))
            if wandb_logger is not None:
                wandb_logger.log({
                'Train/Train Loss CE': losses_ce.avg,
                'Train/Train Loss SCL': losses_scl.avg,
                'Train/Acc@1': top1.avg.item(),
                'Process/Step': i,
                'Process/Epoch': epoch
                })


    return losses.avg, top1.avg



def validate(val_loader, model, criterions):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses_ce = AverageMeter('Loss_ce', ':.5f')
    # losses = AverageMeter('Loss', ':.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses_ce, top1])

    # used for classification_report
    y_true = []
    y_pred = []


    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            images = images.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)
            target = target.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)

            _, output = model(images)

            # concat multi-gpu data
            output = concat_all_gather(output)
            target = concat_all_gather(target)

            # measure accuracy and record loss
            loss_ce = criterions['criterion_ce'](output, target)
            # losses_ce.update(loss_ce.item(), output.size(0))
            # loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, cfg.num_classes)))
            losses_ce.update(loss_ce.item(), output.size(0))
            # losses.update(loss.item(), output.size(0))
            top1.update(acc1[0], output.size(0))

            # for classification_report
            y_true.extend(target.cpu().to(torch.int64).numpy())
            _, preds = output.topk(1, 1, True, True)
            y_pred.extend(preds.cpu().to(torch.int64).numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.print_freq == 0:
                logging.info(progress.display(i))

        # TODO: this should also be done with the ProgressMeter
        logging.info(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

        # report = classification_report(y_true, y_pred, target_names=['0_活体', '1_攻击'], output_dict=True)
        report = classification_report(y_true, y_pred, output_dict=True)
        logging.info('{}'.format(report))

    return losses_ce.avg, top1.avg
    # return losses.avg, top1.avg


def save_checkpoint(state, save_dir ,filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)


def adjust_learning_rate(optimizer, global_step, epoch):
    """Decay the learning rate based on schedule"""
    lr = cfg.learning_rate
    if global_step < cfg.warmup_steps:
        minlr = lr * 0.01
        lr = minlr + (lr - minlr) * global_step / (cfg.warmup_steps - 1)
    else:
        if cfg.cos:
            lr *= 0.5 * (1. + math.cos(math.pi * (global_step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)))
        else:  # stepwise lr schedule
            milestones = cfg.schedule.split(',')
            milestones = [int(milestone) for milestone in milestones]
            for milestone in milestones:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




if __name__ == '__main__':

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    main()
