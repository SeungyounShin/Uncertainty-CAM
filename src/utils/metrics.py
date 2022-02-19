import torch


def topk_accuracy(outputs, labels, topk=(1,)):
    """Computes the accuracy for the top k predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = torch.topk(outputs, k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        topk_accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=False)
            topk_accuracies.append(correct_k.mul_(1.0 / batch_size).item())
        return topk_accuracies

