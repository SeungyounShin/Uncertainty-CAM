import os
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.utils import util, metrics
from src.utils.util import *
from src.core.criterions import mln_uncertainties
from src.builders import model_builder, dataloader_builder, checkpointer_builder,\
                         optimizer_builder, criterion_builder, scheduler_builder,\
                         meter_builder, localizer_builder, evaluator_builder


class BaseEngine(object):

    def __init__(self, config_path, logger, save_dir, device):
        # Assign a logger
        self.logger = logger

        # Load configurations
        config = util.load_config(config_path)

        self.config = config
        self.model_config = config['model']
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.data_config = config['data']

        # Determine which device to use
        device = device  if torch.cuda.is_available() else 'cpu'
        # device = 'cuda:3'
        self.device = torch.device(device)

        if device == 'cpu':
            self.logger.warn('GPU is not available.')
        else:
            self.logger.warn('GPU is available.')

        # Determine the eval standard
        self.eval_standard = self.eval_config['standard']

        # Load a summary writer
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir=self.save_dir)

    def run(self):
        pass


class Engine(BaseEngine):

    def __init__(self, config_path, logger, save_dir, device):
        super(Engine, self).__init__(config_path, logger, save_dir, device)

    def define(self):
        # Build a dataloader
        self.uncertainty_mask = False
        self.logger.info("Build a dataloader")
        self.dataloaders = dataloader_builder.build(
            self.data_config, self.logger)

        self.logger.info("Build a model")
        # Build a model`
        self.model = model_builder.build(
            self.model_config, self.logger, True)
        try:
            self.logger.info("Train with Uncertainty mask")
            self.model_config2 = self.config['model2']
            self.uncertainty_mask = True
            ## model2 is just ResNet
            self.model2 = model_builder.build(
                self.model_config2, self.logger, False)
        except:
            self.logger.warn("Without Uncertainty Mask - Train only MLN")
            self.uncertainty_mask = False

        # Use multi GPUs if available
        if torch.cuda.device_count() > 1:
            self.logger.info("Using multi-GPU")
            self.model = util.DataParallel(self.model)
            if self.uncertainty_mask:
                self.model2 = util.DataParallel(self.model2)
        self.model.to(self.device)
        if self.uncertainty_mask:
            self.model2.to(self.device)

        self.logger.info("Build a Optimizer, Scheduler etc...")
        # Build an optimizer, scheduler and criterion
        self.optimizer = optimizer_builder.build(
            self.train_config['optimizer'], self.model.parameters(), self.logger)

        self.scheduler = scheduler_builder.build(
            self.train_config, self.optimizer, self.logger)
        if self.uncertainty_mask:
            self.criterion = criterion_builder.build(
                self.train_config, self.model_config, self.data_config, self.device, self.logger)
        else:
            self.criterion = criterion_builder.build(
                self.train_config, self.model_config, self.data_config, self.device, self.logger)
        self.loss_meter = meter_builder.build(
            len(self.dataloaders['train']), self.logger)

        # Build a checkpointer
        if self.uncertainty_mask:
            self.checkpointer2 = checkpointer_builder.build(
                self.save_dir, self.logger, self.model2, self.optimizer,
                self.scheduler)

        self.checkpointer = checkpointer_builder.build(
            self.save_dir, self.logger, self.model, self.optimizer,
            self.scheduler)
        checkpoint_path = self.model_config.get('checkpoint_path', '')
        self.misc = self.checkpointer.load(checkpoint_path)

        # Build a localizer and evaluator
        if self.uncertainty_mask:
            self.logger.info("Localizer for Uncertainty Mask")
            self.localizer = localizer_builder.build(
                self.model, MaceCriterion(1000, self.device, False), self.eval_config, self.logger, self.device)
        else:
            self.localizer = localizer_builder.build(
                self.model, self.criterion, self.eval_config, self.logger, self.device)
        self.evaluators = evaluator_builder.build(
            self.eval_config, self.data_config, self.logger)

    def run(self):
        #self.model_config['checkpoint_path'] = './ckpt_imagenet_noise/checkpoint_best_.pth'
        self.define()

        start_epoch, num_steps = 0, 0
        num_epochs = self.train_config.get('num_epochs', 10)
        best_val_metric = 0.

        self.logger.info(
            'Train for {} epochs starting from epoch {}'.format(num_epochs, start_epoch))

        for epoch in range(start_epoch, start_epoch + num_epochs):
            # training phase
            train_start = time.time()
            if self.uncertainty_mask:
                num_steps = self._train_one_epoch_uncertainty_mask(epoch, num_steps)
            else:
                num_steps = self._train_one_epoch(epoch, num_steps)
            train_time = time.time() - train_start

            lr = self.scheduler.get_lr()[0]

            self.logger.infov(
                '[Epoch {}] with lr: {:5f} completed in {:3f} - train loss: {:4f}'\
                .format(epoch, lr, train_time, self.loss_meter.loss))
            self.writer.add_scalar('Train/learning_rate', lr, global_step=num_steps)

            self.scheduler.step()
            self.loss_meter.reset()

            # validation phase
            torch.cuda.empty_cache()

            if epoch%10==0:
                if self.uncertainty_mask:
                    val_metric = self._validate_once_uncertainty_mask(epoch)
                else:
                    val_metric = self._validate_once(epoch)
                self.logger.infov('[Epoch {}] classification performance: {:4f}'.format(epoch, val_metric))

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    if self.uncertainty_mask:
                        self.checkpointer2.save(epoch, num_steps, val_metric=val_metric)
                    else:
                        self.checkpointer.save(epoch, num_steps, val_metric=val_metric)
                #self.checkpointer.save(epoch, num_steps, val_metric=0)

        # test phase
        test_metric = self._test_once()


    def _train_one_epoch(self, epoch, num_steps):
        dataloader = self.dataloaders['train']

        correct = 0
        train_num = 0
        self.model.train()
        for i, (images, labels) in enumerate(dataloader):
            batch_size = images.shape[0]
            images = images.to(self.device)
            labels = labels.to(self.device)
            #masks = masks.to(self.device)

            # Forward propagation
            output_dict = self.model(images)

            # multi class to single class
            #labels[:,0] = 0
            """
            for b in range(batch_size):
                labels[b][0] = 0
                index_one = torch.where(labels[b]==1)[0].cpu().numpy()
                if(len(index_one) == 0):
                    labels[b][0] = 1
                    continue
                choice = np.random.choice(index_one)
                labels[b] = 0
                labels[choice] = 1
            """
            # VOC background to zero
            #labels[:,0] = 0.

            output_dict['labels'] = labels
            #print(output_dict['sigma'])

            # Compute losses
            losses = self.criterion(output_dict)
            mace_loss = losses['mace_avg']
            alea_reg = losses['alea_avg']
            epis_reg = losses['epis_avg']

            loss = losses['mace_avg'] - losses['epis_avg'] #+  3*losses['alea_avg']

            preds = util.mln_gather(output_dict)['mu_sel']
            correct += torch.sum(torch.argmax(preds,dim=-1) == labels).cpu()
            train_num += labels.size(0)
            acc = torch.sum(torch.argmax(preds,dim=-1) == labels).cpu()/float(labels.size(0))
            #acc = 0

            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print losses
            meter_dict = {'mace' : mace_loss.item(), 'alea' : alea_reg.item(), 'epis':epis_reg.item(),
                          'loss' : loss.item(), 'acc' : acc}
            self.loss_meter.update(meter_dict, batch_size)
            self.loss_meter.print_log(meter_dict, epoch, i)
            num_steps += batch_size

        self.writer.add_scalar(
            'Train/loss', self.loss_meter.loss, global_step=num_steps)
        return num_steps

    def _train_one_epoch_uncertainty_mask(self,epoch,num_steps):
        dataloader = self.dataloaders['train']

        correct = 0
        train_num = 0
        self.model.train()
        self.model2.train()
        for i, (images, labels) in enumerate(dataloader):

            batch_size = images.shape[0]
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Get Alea Activation Map
            self.localizer.register_hooks()
            self.localizer.model_ext.loss_type= 'alea_avg'
            cams = self.localizer.localize(torch.tensor(images).to(self.device).float() )
            self.localizer.remove_hooks()
            cams = cams.cpu().detach().squeeze().numpy()

            torch.cuda.empty_cache()
            # Get Blurred Images
            origin_img = images.permute(0,2,3,1).detach().cpu()*torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            origin_img = origin_img.cpu()
            AleaBlured = getBlurAlea(cams, origin_img, ratio=7, mask_ratio=0.4)
            AleaBlured = torch.stack(AleaBlured).cpu().permute(0,3,1,2).float()

            # Forward propagation
            pred = self.model2(AleaBlured.to(self.device))
            output_dict = dict()
            output_dict['preds'] = pred
            output_dict['labels'] = labels

            # Compute losses
            losses = self.criterion(output_dict)

            loss = losses['loss']

            preds = output_dict['preds']
            correct += torch.sum(torch.argmax(preds,dim=-1) == labels).cpu()
            train_num += labels.size(0)
            acc = torch.sum(torch.argmax(preds,dim=-1) == labels).cpu()/float(labels.size(0))
            #acc = 0

            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print losses
            meter_dict = {'loss' : loss.item(), 'acc' : acc}
            self.loss_meter.update(meter_dict, batch_size)
            self.loss_meter.print_log(meter_dict, epoch, i)
            num_steps += batch_size

        self.writer.add_scalar(
            'Train/loss', self.loss_meter.loss, global_step=num_steps)


    def _validate_once_uncertainty_mask(self, epoch):
        dataloader = self.dataloaders['val']
        num_batches = len(dataloader)

        self.model.eval()
        self.localizer.register_hooks()
        epis = list()
        alea = list()
        correct = list()
        valid_num =0

        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward propagation
                output_dict = self.model(images)
                unc_out = mln_uncertainties(output_dict['pi'].detach().cpu(),output_dict['mu'].detach().cpu(),output_dict['sigma'].detach().cpu())
                #print(unc_out)

                #cams = self.localizer.localize(images, labels)

                # Accumulate the classification results
                batch_size = images.size(0)
                preds = util.mln_gather(output_dict)['mu_sel']

                epis.append(unc_out['epis'].cpu())
                alea.append(unc_out['alea'].cpu())

                correct += list((torch.argmax(preds,dim=-1) == labels).cpu().numpy())
                valid_num += labels.size(0)
                #correct.append(torch.max(preds.detach().cpu(),dim=-1)[-1].cpu() == labels.cpu())

                # VOC background to zero
                #labels[:,0] = 0.

                self.evaluators['classification'].accumulate(preds, labels)

                self.logger.info('[Epoch {}] Evaluation batch {}/{}'.format(
                    epoch, i+1, num_batches))

        epis = torch.cat(epis)
        alea = torch.cat(alea)
        #correct = torch.cat(correct)

        # Compute the classification result
        metric = self.evaluators['classification'].compute()
        self.evaluators['classification'].reset()
        self.localizer.remove_hooks()

        meter_dict = {'acc' : sum(correct)/len(correct)}
        self.loss_meter.print_log(meter_dict, epoch, i)

        #figure save
        """
        plt.clf()
        plt.subplot(2,2,1)
        plt.title('alea')
        plt.hist(alea[correct==1].numpy(),bins=20,color='blue',alpha=0.5)
        plt.hist(alea[correct==0].numpy(),bins=20,color='red',alpha=0.5)
        plt.subplot(2,2,2)
        plt.title('epis')
        plt.hist(epis[correct==1].numpy(),bins=20,color='blue',alpha=0.5)
        plt.hist(epis[correct==0].numpy(),bins=20,color='red',alpha=0.5)
        plt.savefig('./ckpt/misclassified.png', dpi=300)
        """

        return metric

        def _validate_once(self, epoch):
            dataloader = self.dataloaders['val']
            num_batches = len(dataloader)

            self.model.eval()
            self.localizer.register_hooks()
            correct = list()
            valid_num =0

            with torch.no_grad():
                for i, (images, labels) in enumerate(dataloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward propagation
                    preds = self.model2(images)

                    batch_size = images.size(0)

                    correct += list((torch.argmax(preds,dim=-1) == labels).cpu().numpy())
                    valid_num += labels.size(0)
                    #correct.append(torch.max(preds.detach().cpu(),dim=-1)[-1].cpu() == labels.cpu())

                    # VOC background to zero
                    #labels[:,0] = 0.

                    self.evaluators['classification'].accumulate(preds, labels)

                    self.logger.info('[Epoch {}] Evaluation batch {}/{}'.format(
                        epoch, i+1, num_batches))

            # Compute the classification result
            metric = self.evaluators['classification'].compute()
            self.evaluators['classification'].reset()
            self.localizer.remove_hooks()

            meter_dict = {'acc' : sum(correct)/len(correct)}
            self.loss_meter.print_log(meter_dict, epoch, i)

            return metric


    def _test_once(self):
        dataloader = self.dataloaders['test']
        num_batches = len(dataloader)

        self.model.eval()
        self.localizer.register_hooks()
        for i, (images, labels, gt_boxes) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward propagation
            output_dict = self.model(images)
            cams = self.localizer.localize(images, labels)

            # Accumulate the classification results
            batch_size = images.size(0)
            preds = util.mln_gather(output_dict)['mu_sel']
            top1_acc, top5_acc = metrics.topk_accuracy(
                preds, labels, topk=(1, 5))
            self.evaluators['classification'].accumulate(top1_acc, batch_size)
            self.evaluators['classification'].accumulate(top5_acc, batch_size)

            # Extract bounding boxes
            unnormalized_images = util.unnormalize_images(images)
            bboxes, blended_bboxes = util.extract_bbox(
                unnormalized_images, cams, gt_boxes, loc_threshold=0.2)

            self.logger.info('[Test] batch {}/{}'.format(i+1, num_batches))
            if (i+1) % 100 == 0:
                self.writer.add_images('Test/{}th images'.format(i+1), blended_bboxes)

        # Compute the classification results
        metric = self.evaluators['top1_cls'].compute()
        self.evaluators['top1_cls'].reset()
        self.evaluators['top5_cls'].reset()
        self.localizer.remove_hooks()
        torch.cuda.empty_cache()
        return metric

    def forward_dataset(self, mode):
        dataloader = self.dataloaders[mode]
        num_batches = len(dataloader)

        self.model.eval()
        epis_ = list()
        alea_ = list()

        for i, (images, labels, gt_boxes) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            output_dict = self.model(images)
            pi, mu, sigma = output_dict['pi'],output_dict['mu'],output_dict['sigma']
            unc_out = mln_uncertainties(pi,mu,sigma)
            (epis, alea) = (
                list(unc_out['epis'].detach().cpu().numpy()),
                list(unc_out['alea'].detach().cpu().numpy())
            )
            epis_ += epis
            alea_ += alea

        return {'epis' : epis_, 'alea':alea_}


    def visualization(self, mode, batch_indices):
        dataloader = self.dataloaders[mode]
        num_batches = len(dataloader)

        self.model.eval()
        self.localizer.register_hooks()
        cams_list = []
        blended_bboxes_list = []
        input_images = []
        for i, (images, labels, gt_boxes) in enumerate(dataloader):
            if i not in batch_indices:
                self.logger.info('[Vis] batch {}/{} skipped'.format(i, num_batches-1))
                continue

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward propagation
            cams = self.localizer.localize(images, labels)

            # Extract bounding boxes
            unnormalized_images = util.unnormalize_images(images)
            bboxes, blended_bboxes = util.extract_bbox(
                unnormalized_images, cams, gt_boxes, loc_threshold=0.2)

            cams_list.append(cams)
            blended_bboxes_list.append(blended_bboxes)
            input_images.append(images)

            self.logger.info('[Vis] batch {}/{} completed'.format(i, num_batches-1))

        self.localizer.remove_hooks()
        vis_output = {
            'cams':torch.cat(cams_list),
            'blended_bboxes': torch.cat(blended_bboxes_list),
            'images': torch.cat(input_images)
        }
        return vis_output
