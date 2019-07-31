# Loss for G in semi supervised setting
import torch
#import torch.functional as F

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
    print("passed")

class ComplementCrossEntropyLoss(torch.nn.Module):
  # Note: This is the cross entropy of the sum of all probabilities of other indices, except for the 
  # This is used in Bayesian GAN semi-supervised learning
  def __init__(self, except_index=None, weight=None, ignore_index=-100, reduction='mean'):
    super(ComplementCrossEntropyLoss, self).__init__()
    self.except_index = except_index
    self.weight = weight
    self.ignore_index = ignore_index
    self.reduction = reduction

  def forward(self, input, target=None):
    # Use target if not None, else use self.except_index
    if target is not None:
      print("called assert")
      _assert_no_grad(target)
    else:
      assert self.except_index is not None
      target = torch.autograd.Variable(torch.LongTensor(input.data.shape[0]).fill_(self.except_index).cuda())
      result = torch.nn.functional.nll_loss(
      torch.log(1. - torch.nn.functional.softmax(input,dim=1) + 1e-4),
      target, weight=self.weight, 
      reduction=self.reduction, 
      ignore_index=self.ignore_index)
    return result