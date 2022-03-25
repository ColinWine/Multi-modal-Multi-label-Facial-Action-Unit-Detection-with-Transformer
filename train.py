import warnings
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from models import *
from dataloader import Aff2CompDataset, SubsetSequentialSampler, SubsetRandomSampler, Prefetcher
from tqdm import tqdm
import os
import time
from sklearn.metrics import f1_score, accuracy_score
from metrics import AccF1Metric, CCCMetric, MultiLabelAccF1
from collections import defaultdict
import opts
from utils import setup_seed, save_checkpoint, AverageMeter
import random
import logging
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


@torch.no_grad()
def evaluate(model, loader, loader_iter, device, num_step=1000):
    model.eval()
    bar = tqdm(range(int(num_step)), desc=f'Validation, {model.task}', colour='green', position=0, leave=False)
    metric_ex = AccF1Metric(ignore_index=7)
    metric_va = CCCMetric(ignore_index=-5.0)
    metric_au = MultiLabelAccF1(ignore_index=-1)
    total_loss = 0
    scores = defaultdict()
    for step in bar:
        t1 = time.time()
        try:
            data = next(loader_iter)
        except StopIteration as e:
            print(e)
            loader_iter = iter(loader)
            break
        t2 = time.time()
        data_time = t2 - t1
        label_ex = data['EX'].long().to(device)
        label_ex[label_ex == -1] = 7
        labels = {
            'VA': data['VA'].float().to(device),
            'AU': data['AU'].float().to(device),
            'EX': label_ex,
        }
        x = {}
        for modality in data:
            x[modality] = data[modality].to(device)
        result = model(x)  # batchx22 12 + 8 + 2
        logits_ex = result[:, 12:19]
        logits_au = result[:, :12]
        logits_va = result[:, 19:21] #tanh??
        if model.task.lower() == 'ex':
            loss = model.get_ex_loss(result, labels['EX'])
        elif model.task.lower() == 'au':
            loss = model.get_au_loss(result, labels['AU'])
        elif model.task.lower() == 'va':
            loss = model.get_va_loss(result, labels['VA'])
        else:
            losses = model.get_mt_loss(result, labels)
            loss = losses[0] + losses[1] + losses[2]
        total_loss += loss.item()

        pred = torch.argmax(logits_ex, dim=1).detach().cpu().numpy().reshape(-1)
        label = label_ex.detach().cpu().numpy().reshape(-1)

        metric_ex.update(pred, label)
        metric_va.update(y_pred=torch.tanh(logits_va).detach().cpu().numpy(), y_true=labels['VA'].detach().cpu().numpy())
        metric_au.update(y_pred=np.round(torch.sigmoid(logits_au).detach().cpu().numpy()), y_true=labels['AU'].detach().cpu().numpy())

        acc_ex = accuracy_score(y_true=label, y_pred=pred)
        bar.set_postfix(data_fetch_time=data_time, batch_loss=loss.item(), avg_loss=total_loss / (step + 1), acc=acc_ex)

    acc_ex, f1_ex = metric_ex.get()
    acc_au, f1_au = metric_au.get()
    scores['EX'] = {'EX:acc': acc_ex, 'f1': f1_ex, 'score': 0.67 * f1_ex + 0.33 * acc_ex}
    scores['AU'] = {'AU:acc': acc_au, 'f1': f1_au, 'score': 0.5 * f1_au + 0.5 * acc_au}
    scores['VA'] = {'VA:ccc_v': metric_va.get()[0],'ccc_a': metric_va.get()[1], 'score': metric_va.get()[2]}
    model.train()
    metric_va.clear()
    metric_au.clear()
    metric_ex.clear()
    return scores, loader_iter


def train(args, model, dataset, optimizer, epochs, device):
    early_stopper = EarlyStopper(num_trials=args['early_stop_step'], save_path=f'{args["checkpoint_path"]}/best.pth')
    downsample_rate = args.get('downsample_rate')
    downsample = np.zeros(len(dataset), dtype=int)
    downsample[np.arange(0, len(dataset) - 1, downsample_rate)] = 1
    start_epoch = 0
    if args['resume'] == True:
        start_epoch = args['start_epoch']
    learning_rate = args['learning_rate']
    for epoch in range(start_epoch,epochs):
        if epoch == 30:
            learning_rate = learning_rate*0.1
        if epoch == 60:
            learning_rate = learning_rate*0.1

        random.shuffle(downsample)
        dataset.set_aug(True)
        train_sampler = SubsetSequentialSampler(np.nonzero(dataset.train_ids*downsample)[0], shuffle=True)
        train_loader = DataLoader(dataset, batch_size=args['batch_size'], sampler=train_sampler, num_workers=0,
                                pin_memory=False,
                                drop_last=True)
        
        print('Training set length: ' + str(sum(dataset.train_ids*downsample)))
        bar = tqdm(train_loader, desc=f'Training {model.task}, Epoch:{epoch}', colour='blue', position=0, leave=True)
        logging.info(f'Training {model.task}, Epoch:{epoch}')
        t1 = time.time()
        total_loss, ex_loss_record,au_loss_record,va_loss_record = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        prefetcher = Prefetcher(bar)
        data = prefetcher.next()
        step = -1
        while data is not None:
            step += 1
            t2 = time.time()
            data_time = t2 - t1
            optimizer.zero_grad()
            label_ex = data['EX'].long().to(device)
            label_ex[label_ex == -1] = 7
            labels = {
                'VA': data['VA'].float().to(device),
                'AU': data['AU'].float().to(device),
                'EX': label_ex,
            }

            # ids = data['Index'].long()
            x = {}
            for modality in data:
                x[modality] = data[modality].to(device)
                #x['clip'] = data['clip'].to(device)
                #x['audio_features'] = data['audio_features'].to(device)
            result = model(x)  # batchx22 12 + 8 + 2
            if model.task.lower() == 'ex':
                loss = model.get_ex_loss(result, labels['EX'])
            elif model.task.lower() == 'au':
                loss = model.get_au_loss(result, labels['AU'])
            elif model.task.lower() == 'va':
                loss = model.get_va_loss(result, labels['VA'])
            else:
                losses = model.get_mt_loss(result, labels, normalize = False)
                loss = 3*losses[0] + losses[1] + losses[2]
                ex_loss_record.update(losses[0].item())
                au_loss_record.update(losses[1].item())
                va_loss_record.update(losses[2].item())

            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())
            if model.task.lower() == 'all':
                bar.set_postfix(total = total_loss.avg, ex=ex_loss_record.avg, au=au_loss_record.avg, va=va_loss_record.avg)
            else:
                bar.set_postfix(data_fetch_time=data_time, batch_loss=loss.item(), avg_loss=total_loss.avg)
            
            t1 = time.time()
            data = prefetcher.next()
        logging.info(f'Total Loss,{total_loss.avg}, Ex:{ex_loss_record.avg}, AU:{au_loss_record.avg}, VA:{va_loss_record.avg}')

        save_checkpoint(state=model.state_dict(), filepath=args["checkpoint_path"], filename='latest.pth')
        #if step % eval_step == 0 and step != 0:
        dataset.set_aug(False)
        val_sampler = SubsetSequentialSampler(np.nonzero(dataset.val_ids*downsample)[0], shuffle=True)
        val_loader = DataLoader(dataset, batch_size=args['batch_size'] * 4, sampler=val_sampler, num_workers=0,
                                pin_memory=False,
                                drop_last=True)
        print('Validation set length: ' + str(sum(dataset.val_ids*downsample)))
        val_loader_iter = iter(val_loader)
        scores, val_loader_iter = evaluate(model, val_loader, val_loader_iter, device,
                                            num_step=int(sum(dataset.val_ids*downsample)/(args['batch_size']*4)))
        score_str = ''
        if model.task == 'ALL':
            total_score = 0
            for task in ['EX','AU','VA']:
                score_dict = scores[task]
                for k, v in score_dict.items():
                    score_str += f'{k}:{v:.3},'
                total_score = total_score + score_dict["score"]
        else:
            score_dict = scores[model.task]
            for k, v in score_dict.items():
                score_str += f'{k}:{v:.3}, '
            total_score = score_dict["score"]
        print(f'Training,{args["task"]}, Epoch:{epoch}, {score_str}')
        logging.info(f'Training,{args["task"]}, Epoch:{epoch}, {score_str}')
        if not early_stopper.is_continuable(model, total_score):
            print(f'validation: best score: {early_stopper.best_accuracy}')
            logging.info(f'validation: best score: {early_stopper.best_accuracy}')
            break


def main(args):
    setup_seed(args.get('seed'))
    task = args.get('task')
    print(f'Task: {task}')
    print('Model:',opt['model_name'])
    print('Modality:',opt['modality'])
    print('clip size',opt['n_frames'],opt['image_size'])
    log_file_name = opt['model_name']+'_'+opt['modality']+'_log.txt'
    logging.basicConfig(filename=os.path.join(args['exp_dir'],log_file_name), level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger()
    
    # model
    if opt['model_name'] == 'avformer':
        model = TwoStreamAuralVisualFormer(modality=args['modality'], task=task)
    elif opt['model_name'] == 'vformer':
        model = VisualFormer(modality=args['modality'], task=task)
    elif opt['model_name'] == 'vggformer':
        model = VGGVisualFormer(modality=args['modality'], task=task)
    elif opt['model_name'] == 'emonet':
        model = ImageEmoNetModel(modality=args['modality'], task=task)
    elif opt['model_name'] == 'tformer':
        model = SpatialTemporalFormer(modality=args['modality'], task=task)
    elif opt['model_name'] == 'sformer':
        model = SpatialFormer(modality=args['modality'], task=task)
    elif opt['model_name'] == 'dsformer':
        model = DualSpatialFormer(modality=args['modality'], task=task)
    elif opt['model_name'] == 'i3d':
        model = VisualI3DModel(modality=args['modality'], task=task)
    elif opt['model_name'] == 'mc3d':
        model = VisualMC3DModel(modality=args['modality'], task=task)
    elif opt['model_name'] == 'van':
        model = SpatialVAN(modality=args['modality'], task=task)
    elif opt['model_name'] == 'audio':
        model = Audio_only(modality=args['modality'], task=task)
    else:
        model = ImageResNetModel(task)

    modes = model.modes
    model = model.to(torch.cuda.current_device())



    args['checkpoint_path'] = os.path.join(args['exp_dir'], 'pretrain')
    if args['resume'] and os.path.exists(f'{args["checkpoint_path"]}/latest.pth'):
        print('Loading weight from:{}'.format(f'{args["checkpoint_path"]}/latest.pth'))
        pretrained_dict = torch.load(f'{args["checkpoint_path"]}/latest.pth')
        model.load_state_dict(pretrained_dict,strict= False)
    model.train()

    # load dataset (first time this takes longer)
    dataset = Aff2CompDataset(args)

    dataset.set_modes(modes)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    #train(args, model, train_loader, val_loader, optimizer, epochs=args['epochs'], device=torch.cuda.current_device())
    train(args, model, dataset, optimizer, epochs=args['epochs'], device=torch.cuda.current_device())


if __name__ == '__main__':
    opt = opts.parse_opt()
    torch.cuda.set_device(opt.gpu_id)
    opt = vars(opt)
    main(opt)
