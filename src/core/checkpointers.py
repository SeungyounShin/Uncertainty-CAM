import os
import torch
import numpy as np


class CustomCheckpointer(object):
    def __init__(self, save_dir, logger, model, optimizer=None, scheduler=None,
                 standard='accruacy', best_val_metric=-np.inf):
        self.logger = logger
        self.save_dir = save_dir

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.best_val_metric = best_val_metric
        self.standard = standard

        self.logger.infov('Checkpointer is built.')


    def save(self, epoch, num_steps, val_metric):
        #if val_metric <= self.best_val_metric:
        #    return

        checkpoint_path = os.path.join(self.save_dir, 'checkpoint_best.pth')

        model_params = {'epoch': epoch, 'num_step': num_steps}
        #if torch.cuda.device_count() > 1:
        #    model_params['model_state_dict'] = self.model.module.state_dict()
        #else:
        model_params['model_state_dict'] = self.model.state_dict()

        model_params['optimizer_state_dict'] = self.optimizer.state_dict()
        model_params['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save the best checkpoint
        if val_metric >= self.best_val_metric:
            self.best_val_metric = val_metric
            torch.save(model_params, checkpoint_path)
            self.logger.info(
                'The best checkpoint is saved for epoch={}, steps={}.'.format(epoch, num_steps))

        # Update the checkpoint record
        best_checkpoint_info = {
            'epoch': epoch, 'num_step': num_steps, self.standard + '_val': val_metric}
        self._record_checkpoint(best_checkpoint_info, self.save_dir, 'best_checkpoint')

    def load(self, checkpoint_path='', test=False):
        if test:
            checkpoint_path = os.path.join(self.save_dir, 'checkpoint_best.pth')

        if not self._has_checkpoint(checkpoint_path): # Override argument with existing checkpoint
            self.logger.info(
                "A checkpoint - {}, does not exists. Start from scratch.".format(
                checkpoint_path))
            return

        self.logger.info("Loading checkpoint from {}".format(checkpoint_path))
        checkpoint = self._load_checkpoint(checkpoint_path)

        self.model.load_state_dict(
            checkpoint.pop('model_state_dict'), strict=True)

        return checkpoint

    def _freeze(self):
        for param in self.model.layers.parameters():
            param.requires_grad = False

    def _has_checkpoint(self, checkpoint_path):
        return os.path.exists(checkpoint_path)

    def _get_checkpoint_path(self):
        record_path = os.path.join(self.checkpoint_dir, "last_checkpoint")
        try:
            with open(record_path, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            self.logger.warn('If last_checkpoint file doesn not exist, maybe because \
                              it has just been deleted by a separate process.')
            last_saved = ''
        return last_saved

    def _record_checkpoint(self, checkpoint_info, save_dir, file_name):
        record_path = os.path.join(save_dir, file_name)
        with open(record_path, 'w') as f:
            f.write(str(checkpoint_info))

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        checkpoint['checkpoint_path'] = checkpoint_path
        return checkpoint
