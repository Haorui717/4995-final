import datetime
import logging
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import dataloader


from data.classifier_dataset import ClassifierDataset
from option.train_classifier_option import TrainClassifierOpts
from util import tensor2np, save_img
from models.classifier import Classifier


class Trainer:
    def __init__(self, opts):
        self.opts = opts
        self.network = Classifier()
        self.network = self.network.to(opts.device)

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=opts.lr)
        self.loss = nn.BCELoss()
        self.trainset = ClassifierDataset(opts.trainset_path,
                                          opts.image_list_path,
                                          opts.list_attr_celeba_path)
        self.testset = ClassifierDataset(opts.testset_path,
                                          opts.image_list_path,
                                          opts.list_attr_celeba_path)

        self.trainloader = dataloader.DataLoader(self.trainset,
                                                 shuffle=True,
                                                 batch_size=opts.batch_size,
                                                 num_workers=opts.num_workers)
        self.testloader = dataloader.DataLoader(self.testset,
                                                shuffle=False,
                                                batch_size=opts.batch_size,
                                                num_workers=opts.num_workers)
        # self.ckpt_dir = opts.ckpt_dir
        self.cls_idx = opts.cls_idx
        self.attr_name = self.trainset.attrs[self.cls_idx]
        self.best_loss = None

        os.makedirs(opts.log_dir, exist_ok=True)
        handlers = [logging.FileHandler(os.path.join(opts.log_dir, 'train_classifier '+str(datetime.datetime.now())+'.log')),
                    logging.StreamHandler()]
        logging.basicConfig(handlers=handlers,
                            format='%(asctime)s %(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def train(self):
        size = len(self.trainset)
        self.network.train()
        for i in range(self.opts.num_epochs):
            for batch, (x, label) in tqdm(enumerate(self.trainloader)):
                x, label = x.to(self.opts.device), label.to(self.opts.device)
                label = label[:, self.cls_idx].squeeze().float()
                y = self.network(x)
                y = y.squeeze()
                loss = self.loss(y, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (batch + 1) % 50 == 0:
                    loss, current = loss.item(), (batch+1)*len(x) + i*size
                    self.logger.info(f'loss: {loss:>7f}  [{current:>5d}/{size * self.opts.num_epochs:>5d}]')

                if ((batch + 1) * x.shape[0] + i*size) % self.opts.num_eval == 0:
                    self.logger.info('evaluate')
                    result = self.evaluate()

                if ((batch + 1) * x.shape[0] + i*size) >= self.opts.num_train:
                    return

    def evaluate(self):
        self.network.eval()
        size = len(self.testset)
        num_batch = len(self.testloader)
        tot_loss, acc = 0, 0
        with torch.no_grad():
            for batch, (x, label) in enumerate(self.testloader):
                x, label = x.to(self.opts.device), label.to(self.opts.device)
                label = label[:, self.cls_idx].squeeze().float()
                y = self.network(x)
                y = y.squeeze()
                loss = self.loss(y, label)
                tot_loss += loss.item()
                y[y<=0.5], y[y>0.5] = 0, 1
                acc += (y == label).type(torch.float).sum().item()

        tot_loss /= num_batch
        acc /= size
        self.logger.info(f"validation error: Accuracy: {(100 * acc):>0.1f}%, Avg loss: {tot_loss:>8f}")
        print(acc)
        if self.best_loss is None or tot_loss < self.best_loss:
            self.best_loss = tot_loss
            self.save_best_weights()
            self.logger.info(f"save the best weight")

        self.network.train()
        return acc, tot_loss

    def show_samples(self, path):
        network_tmp = Classifier().to(self.opts.device)
        network_tmp.load_state_dict(torch.load(path))
        network_tmp.eval()
        with torch.no_grad():
            for batch, (x, label) in enumerate(self.testloader):
                if batch == 25: break
                x, label = x.to(self.opts.device), label.to(self.opts.device)
                label = label[:, self.cls_idx].squeeze().float()
                y = network_tmp(x)
                y = y.squeeze()
                for i in range(x.shape[0]):
                    img = x[i]
                    img_np = tensor2np((img+1)/2)
                    pred = 0 if y[i] < 0.5 else 1
                    if pred == 0:
                        save_img('samples/0', f'{batch}_{i}_{label[i] < 0.5}.png', img_np)
                    else:
                        save_img('samples/1', f'{batch}_{i}_{label[i] > 0.5}.png', img_np)

    def save_best_weights(self):
        os.makedirs(self.opts.ckpt_dir, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(self.opts.ckpt_dir, f'{self.attr_name}.pt'))

    def train_all_attr(self):
        summary = []
        for cls_idx in range(self.opts.start, self.opts.end):
            self.reset_network()
            self.cls_idx = cls_idx
            self.attr_name = self.trainset.attrs[self.cls_idx]
            self.train()
            acc, loss = self.evaluate()
            summary.append([self.attr_name, acc, loss])
            self.logger.critical(f"{self.attr_name}: acc: {acc}, loss: {loss}")

        for i in summary:
            self.logger.critical(f"{i[0]}: acc: {i[1]}, loss: {i[2]}")

    def reset_network(self):
        self.network = Classifier()
        self.network = self.network.to(self.opts.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=opts.lr)
        self.best_loss = None

if __name__ == '__main__':
    opts = TrainClassifierOpts().parse()
    trainer = Trainer(opts)
    # trainer.train()
    # trainer.show_samples()
    trainer.train_all_attr()
