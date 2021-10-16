import inspect
import torch
import importlib
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from  torchmetrics.functional import auroc
import random

from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve
from pytorch_lightning.metrics import ConfusionMatrix


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        # print(alpha)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, labels, filename = batch
        out = self(img)


        loss = self.loss_function(out, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels, filename = batch
        out = self(img)

        loss = self.loss_function(out, labels)

        if labels.dim() > 1:
            label_digit = labels.argmax(axis=1)
        else:
            label_digit = labels

        out_digit = out.argmax(axis=1)

        correct_num = sum(label_digit == out_digit).cpu().item()

        try:
            auc = auroc(out,label_digit,num_classes=self.hparams.class_num)
        except ValueError:
            auc = 0.5

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/len(out_digit),
                 on_step=False, on_epoch=True, prog_bar=True)
        
        return {'pred': out, 'target': labels}
        # return (correct_num, len(out_digit))

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['pred'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])

        confmat = ConfusionMatrix(num_classes=2)
        print(confmat(preds.cpu(), targets.cpu()))
    
    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
        

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'focal':
            self.loss_function = FocalLoss(self.hparams.class_num, alpha = torch.tensor([0.4, 0.6]).cuda(), gamma = 4)
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def confusion(self, y_true, y_pred, mode, logPath):
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        confmat = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(2.5, 2.5))

        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center', fontsize=10)
        
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        Accuracy = (TP+TN)/(TP+FP+FN+TN)
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        Sensitivity = TP / (TP + FN)


        plt.xlabel('Predict', fontsize=10)        
        plt.ylabel('True', fontsize=10)
        
        plt.title(str(mode)+' Accuracy : {:.2f} | Specificity : {:.2f} | Sensitivity : {:.2f}'.format(Accuracy, Specificity, Sensitivity), fontsize=10)
        plt.savefig(logPath+"//"+str(mode)+"_confusion .jpg", bbox_inches='tight')
        plt.close('all')
        # plt.show()
        # return Accuracy
