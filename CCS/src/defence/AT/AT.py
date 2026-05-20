import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from src.defence.AT.Patch import SingleAnchorDPR, random_location,get_masks
def max_margin_loss(out, iden, threshold=None):
    real = out.gather(1, iden.unsqueeze(1)).squeeze(1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == iden, tmp1[:, -2], tmp1[:, -1])
    margin = out.gather(1, new_y.unsqueeze(1)).squeeze(1)
    if threshold is None:
        return (-1 * real).mean() + margin.mean()
    else:
        return (-1 * real).mean() + torch.abs(margin - threshold).mean()

def perturb_input(model,
                  x_natural,
                  y_natural,
                  step_size=0.0035,
                  epsilon=0.031,
                  distance='l_inf',
                  device=torch.device("cuda:0")
                  ):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        # y_natural = model(x_natural)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # loss_kl = F.kl_div(F.log_softmax(model(normalizer(x_adv)), dim=1),
                #                    F.softmax(y_natural, dim=1),
                #                    reduction='sum')
                # loss_oh = F.cross_entropy(model(normalizer(x_adv,device)),y_natural.max(1)[1],reduction='sum')
                loss_mm = max_margin_loss(model(x_adv), y_natural.to(device))
            grad = torch.autograd.grad(loss_mm, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)
        optimizer_delta = torch.optim.SGD([delta], lr=epsilon / perturb_steps * 2)
        for _ in range(perturb_steps):
            adv = x_natural + delta
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = - max_margin_loss(model(adv), y_natural)
            loss.backward()
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            optimizer_delta.step()
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    elif distance == 'l_inf_patch':
        batch_size, channels, h, w = x_natural.shape
        radius = 10
        centerx = torch.from_numpy(np.random.randint(radius, h - radius, batch_size)).long()
        centery = torch.from_numpy(np.random.randint(radius, w - radius, batch_size)).long()
        num_regions = 36
        tmp = np.ones((batch_size, num_regions)) * radius
        r = torch.from_numpy(tmp)
        masks = SingleAnchorDPR(centerx, centery, r, num_regions, h, w, False)
        masks = masks.unsqueeze(1)
        patches = torch.tensor(np.random.uniform(low=0.0, high=1.0, size=x_natural.shape).astype(np.float32),device=device, requires_grad=True)
        inverse_masks = 1.0 - masks
        x_adv = inverse_masks * x_natural + masks * patches
        x_adv_init = x_adv.detach().clone()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_mm = max_margin_loss(model(x_adv), y_natural.to(device))
            grad = torch.autograd.grad(loss_mm, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_adv_init - epsilon), x_adv_init + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'patch':
        batch_size, channels, h, w = x_natural.shape
        radius = 10
        centerx = torch.from_numpy(np.random.randint(radius, h - radius, batch_size)).long()
        centery = torch.from_numpy(np.random.randint(radius, w - radius, batch_size)).long()
        num_regions = 36
        tmp = np.ones((batch_size, num_regions)) * radius
        r = torch.from_numpy(tmp)
        masks = SingleAnchorDPR(centerx, centery, r, num_regions, h, w, False)
        masks = masks.unsqueeze(1)
        inverse_masks = 1.0 - masks

        patches = torch.tensor(np.random.uniform(low=0.0, high=1.0, size=x_natural.shape).astype(np.float32),device=device)
        x_adv = inverse_masks * x_natural + masks * patches
        for _ in range(perturb_steps):
            x_adv.requires_grad = True
            with torch.enable_grad():
                loss_mm = max_margin_loss(model(x_adv), y_natural.to(device))
            grad = torch.autograd.grad(loss_mm, [x_adv])[0]
            update = step_size * torch.sign(grad.detach())
            x_adv = x_adv.detach() + masks * update
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)  # 归一化到[0, 1]区间
    return x_adv  # 返回生成的对抗样本




