import numpy as np

from tqdm import tqdm
import shutil
import random
import time

import scipy.stats
import math

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn.functional as F

from agents.base import BaseAgent

# import your classes here
from graphs.models.PNC_Test import resnet_50
from graphs.losses.Triplet_global_center import HardMiningLoss

from datasets.birds import Bird
from datasets.cars import Car
from datasets.isc import Isc
from datasets.sop import Sop

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList, cls_accuracy
from utils.misc import print_cuda_statistics
from utils.train_utils import to_one_hot
from utils.utils import evaluate,QG_evaluate
# from utils.center import Center
from graphs.losses.get_Center import Center

cudnn.benchmark = True


class PNC_TestAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        # self.model = resnet50(pretrained=True, num_classes=self.config.train_classes, is_train=True)
        # self.testmodel = resnet50(pretrained=False, num_classes=self.config.train_classes, is_train=False)

        self.model = resnet_50(
            pretrained=True,
            num_classes=self.config.train_classes,
            saliency = None,
            pool_type = "max_avg",
            is_train = True,
            scale = self.config.scale)

        self.testmodel = resnet_50(
            pretrained=False,
            num_classes= self.config.train_classes,
            saliency = None,
            pool_type = "max_avg",
            is_train = False,
            threshold = self.config.threshold,
            scale = self.config.scale)

        # define data_loader
        # self.data_loader = Bird(config=config)
                # define data_loader
        if config.data_loader == 'Birds':
            self.data_loader = Bird(config=config)
        elif config.data_loader == 'Cars':
            self.data_loader = Car(config=config)
        elif config.data_loader == 'Sop':
            self.data_loader = Sop(config=config)
            pass
        elif config.data_loader == 'Isc':
            self.data_loader = Isc(config=config)
            pass
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

        from queue import Queue,LifoQueue,PriorityQueue
        self.q = Queue(maxsize=5)

        # define loss
        self.loss = nn.CrossEntropyLoss()

        self.Center = Center()

        self.triplet = HardMiningLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, 
            weight_decay = self.config.momentum)

        # define scheduler
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5,10,25,55,75,95,500,1000],gamma=5e-1)
        self.decay_time = [False,False]
        self.init_lr = self.config.learning_rate
        self.decay_rate = 0.1

        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 0
        self.best_metric = 0

        self.epoch_loss = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.manual_seed(self.manual_seed)
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.testmodel = self.testmodel.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment="Agent")

    def lr_scheduler(self, epoch):
        if epoch>=0.5*self.config.max_epoch and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.8*self.config.max_epoch and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return


    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'loss': self.loss.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)

        epoch_file_name = str(self.current_epoch) + ".pth.tar"
        torch.save(state, self.config.checkpoint_dir + epoch_file_name)
        
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, file_name="checkpoint.pth.tar"):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.loss.load_state_dict(checkpoint['loss'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        t1 = time.time()
        for epoch in range(self.current_epoch, self.config.max_epoch + 1):
            e1 = time.time()
            is_best=0

            self.epoch_loss = AverageMeter()
            self.top1 = AverageMeter()
            self.top5 = AverageMeter()

            self.lr_scheduler(epoch)

            self.train_one_epoch()

            if epoch % self.config.valid_every == 0:
            # if epoch % self.config.valid_every == 0:
                #valid_acc = self.validate()

                train_acc, valid_acc = self.validate()
                self.summary_writer.add_scalars("epoch/Recall", {"train_acc":train_acc,
                                                                 "valid_acc": valid_acc}, epoch)
                is_best = valid_acc > self.best_metric
                if is_best:
                    self.best_metric = valid_acc
            e2 = time.time()
            self.logger.info("Epoch: {}, time (seconds): {:.2f}.".format(epoch,e2-e1))
            self.save_checkpoint(is_best=is_best)
            self.current_epoch += 1
        t2 = time.time()
        self.logger.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        # self.scheduler.step()

        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            # output = self.model(data)

            target_one_hot = to_one_hot(target.cpu(), num_class=self.config.train_classes).to(self.device)

            # label soomth
            # target_one_hot = (1.0 - 0.1) * target_one_hot + 0.1 / 100
            #target_float_hot = to_float_hot(target.cpu(), num_class=self.config.train_classes).to(self.device)

            # loss = self.loss(output, target)

            inputs_var, labels_var = data, target

            class_label = torch.Tensor(np.array(range(self.config.train_classes)))
            center_labels_var = torch.autograd.Variable(class_label.to(torch.long)).cuda()

            fvec, feature, class_weight = self.model(inputs_var)

            if self.q.full(): 

                self.q.get()
                self.q.put(class_weight.cpu())

                # import pdb
                # pdb.set_trace()
                a = list(self.q.queue)
                
                # temp = (a[0] + a[1] a[2] + a[3] + a[4]) 
                temp = ((a[0] + a[1] + a[2] + a[3] + a[4])/5)
                class_weight = 0.2 * class_weight + temp.to(self.device)


            else: 
                self.q.put(class_weight.cpu())

            

            #on_hot vector
            labels_var_one_hot = target_one_hot
            # inter_class_distance
            fvec = fvec - 4 * labels_var_one_hot.cuda()
            #intra_class_distance
            loss_1 = self.loss(fvec, labels_var)

            origin_class_weight = class_weight
            
            batch_center = self.Center(feature, target, self.config.train_classes, class_weight)
            # batch_center = F.relu(batch_center)
            batch_center = F.normalize(batch_center, p=2,dim=1)

            # if self.current_epoch < 13:
                
            #     linear_beta = (13 - self.current_epoch) / 13

            #     norm_beta = scipy.stats.norm(0, 1).pdf(self.current_epoch/12/2)
                
            #     beta = 1

            #     class_weight = torch.div(class_weight + beta * batch_center, 2)


            class_weight = torch.div(class_weight + batch_center, 2)

            # class_weight = torch.div(class_weight - batch_center, 2)

            class_weight = F.normalize(class_weight)

            center_loss = self.loss(torch.mm(class_weight, torch.t(class_weight)),center_labels_var)

            triplet_loss = self.triplet(feature, target, class_weight)

            triplet_origin_loss = self.triplet(feature, target, origin_class_weight)

            loss = 0.5 * center_loss + loss_1 + 0.1 * triplet_loss
            # loss = 0.5 * center_loss + 0.1 * triplet_loss

            if self.config.loss_mode == '100': 
                loss = triplet_origin_loss
            if self.config.loss_mode == '101': 
                loss = triplet_origin_loss + loss_1
            if self.config.loss_mode == '110': 
                loss = triplet_loss
            if self.config.loss_mode == '011': 
                loss = loss_1
            if self.config.loss_mode == '111': 
                loss = 0.5 * center_loss + loss_1 + 0.1 * triplet_loss

            prec1, prec5 = cls_accuracy(fvec, target, topk=(1, 5))


            self.epoch_loss.update(loss.item())
            self.top1.update(prec1.item())
            self.top5.update(prec5.item()) 

            # loss.backward()
            loss.backward(retain_graph=True)

            self.optimizer.step()



            if batch_idx % self.config.log_interval == 0:
                
                # self.logger.info(f'center_loss:{center_loss}\t loss_1:{loss_1}\t triplet_loss:{triplet_loss}')

                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\tPrec@5 {top5.val:.3f} ({top5.avg:.3f}) \tlr {lr}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss=self.epoch_loss,
                           top1=self.top1, top5=self.top5, lr=self.optimizer.param_groups[0]['lr']))
            self.current_iteration += 1
            if np.isnan(float(loss.item())):
                raise ValueError('Loss is nan during training...')

            self.summary_writer.add_scalar("batch/loss", self.epoch_loss.avg, self.current_iteration)
            self.summary_writer.add_scalar("batch/top1", self.top1.avg, self.current_iteration)
            self.summary_writer.add_scalar("batch/top5", self.top5.avg, self.current_iteration)
            self.summary_writer.add_scalar("batch/lr", self.optimizer.param_groups[0]['lr'], self.current_iteration)



    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.testmodel.load_state_dict(self.model.state_dict())
        self.testmodel.eval()

        gpu_device = self.config.gpu_device

        if self.config.data_loader == 'Isc':
            nmi, train_recall = evaluate(gpu_device, self.testmodel, self.data_loader.train_loader, self.config.train_classes, name='tra_similar.jpg')
            nmi, recall = QG_evaluate(gpu_device, self.testmodel, self.data_loader.query_loader, self.data_loader.gallery_loader, self.config.test_classes, name='tes_similar.jpg')
        else:
            nmi, train_recall = evaluate(gpu_device, self.testmodel, self.data_loader.valid_loader, self.config.train_classes, name='tra_similar.jpg')
            nmi, recall = evaluate(gpu_device, self.testmodel, self.data_loader.test_loader, self.config.test_classes, name='tes_similar.jpg')
        self.logger.info("**Evaluating...**")
        self.logger.info("NMI: {:.3f}".format(nmi * 100))
        if nmi != 0:
            for i, k in enumerate([1, 2, 4, 8]):
                self.logger.info("R@{} : {:.3f}".format(k, 100 * recall[i]))
            return train_recall[0],recall[0]
        else:
            for i, k in enumerate([1, 10, 20, 30]):
                self.logger.info("R@{} : {:.3f}".format(k, 100 * recall[i]))
            return train_recall[0],recall[0]

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
