import torch
import numpy as np
from src.utils import metrics
from sklearn.metrics import average_precision_score

class Evaluator(object):

    def reset(self):
        pass

    def accumulate(self, predictions, gts):
        pass

    def compute(self):
        pass

    def print_log(self, epoch, num_steps):
        pass


class ClsEvaluator(Evaluator):

    def __init__(self, name, logger, topk=(1,)):
        self.name = name
        self.logger = logger
        self.topk = topk
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def accumulate(self, predictions, gts):
        try:
            batch_size = predictions.shape[0]
        except:
            batch_size = 1

        accuracy = metrics.topk_accuracy(predictions, gts, topk=self.topk)[0]
        #print(accuracy)
        self.sum += accuracy * batch_size
        self.count += batch_size

    def compute(self):
        self.avg = self.sum / float(self.count)
        return self.avg


class APEvaluator(Evaluator):

    def __init__(self, name, logger):
        self.name = name
        self.logger = logger
        self.reset()

    def reset(self):
        self.predictions = None
        self.gts = None

    def accumulate(self, predictions, gts):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(gts, torch.Tensor):
            gts = gts.detach().cpu().numpy()

        if self.predictions is None:
            self.predictions = predictions
        else:
            self.predictions = np.concatenate((self.predictions, predictions))

        if self.gts is None:
            self.gts = gts
        else:
            self.gts = np.concatenate((self.gts, gts))

    def compute(self):
        ap = average_precision_score(self.gts, self.predictions, average='micro')
        #precision, recall, threshold = metrics.precision_recall_curve(
        #    np.reshape(self.gts, -1), np.reshape(self.predictions, -1))
        return ap
