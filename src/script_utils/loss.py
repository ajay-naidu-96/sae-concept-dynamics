import torch

def mse_criterion(x, x_hat, pre_codes, codes, dictionary):

  mse = (x - x_hat).square().mean()

  return mse


def mse_l1_criterion(x, x_hat, pre_codes, codes, dictionary, l1_coefficient=1e-3):

    mse_loss = mse_criterion(x, x_hat, pre_codes, codes, dictionary)
    
    sparsity_loss = torch.mean(torch.sum(codes.abs(), dim=1))

    return mse_loss + sparsity_loss * l1_coefficient
    

def reanim_mse_l1_criterion(x, x_hat, pre_codes, codes, dictionary, l1_coefficient=1e-3):

  mse_loss = mse_criterion(x, x_hat, pre_codes, codes, dictionary)

  sparsity_loss = torch.mean(torch.sum(codes.abs(), dim=1))

  mse_loss += sparsity_loss * l1_coefficient

  is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
  reanim_loss = (pre_codes * is_dead[None, :]).mean()

  mse_loss -= reanim_loss * 1e-3

  loss = mse_loss - reanim_loss * 1e-3

  return loss


def reanimate_criterion(x, x_hat, pre_codes, codes, dictionary):

  mse_loss = mse_criterion(x, x_hat, pre_codes, codes, dictionary)

  is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
  reanim_loss = (pre_codes * is_dead[None, :]).mean()

  mse_loss -= reanim_loss * 1e-3

  loss = mse_loss - reanim_loss * 1e-3

  return loss


  

