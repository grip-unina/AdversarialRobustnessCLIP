import torch
import torch.nn as nn
from torchattacks.attack import Attack

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def get_adv_noise(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_noise = torch.zeros_like(images)

        if self.random_start:
            # Starting at a uniformly random point
            adv_noise = adv_noise.uniform_(-self.eps, self.eps)
            if msk is not None:
                adv_noise = msk*adv_noise

        for _ in range(self.steps):
            adv_noise.requires_grad = True
            outputs = self.get_logits(images+adv_noise)

            # Calculate loss
            if self.targeted:
                cost = loss(outputs, target_labels)
            else:
                cost = -loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_noise, retain_graph=False, create_graph=False
            )[0]
            
            grad = grad.sign()
            
            adv_noise = adv_noise.detach() - self.alpha * grad
            adv_noise = torch.clamp(adv_noise, min=-self.eps, max=self.eps)
            
        return adv_noise

def norm_h(x, p=2):
    return torch.norm(x.reshape(x.shape[0], -1), p=p, dim=1).view(-1, 1, 1, 1)

def normalize_b(x, eps_for_division, p=2):
    norms = torch.norm(x.reshape(x.shape[0], -1), p=p, dim=1) + eps_for_division
    return x / norms.view(-1, 1, 1, 1)
    
class PGDL(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2, L1

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 10)
        p: 2 for L2 and 1 for L1. (Default: 2)
        random_start (bool): using random initialization of delta. (Default: True)

    """

    def __init__(
        self,
        model,
        eps=1.0,
        alpha=0.2,
        steps=10,
        p=2,
        random_start=True,
        eps_for_division=1e-15,
    ):
        super().__init__("PGDL", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.p = p
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]

    def get_adv_noise(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_noise = torch.zeros_like(images)

        if self.random_start:
            # Starting at a uniformly random point
            adv_noise = adv_noise.normal_()

            r = torch.zeros_like(norm_h(adv_noise, p=self.p))
            adv_noise = self.eps * r * normalize_b(adv_noise, self.eps_for_division, p=self.p)

        for _ in range(self.steps):
            adv_noise = adv_noise.detach()
            adv_noise.requires_grad = True
            outputs = self.get_logits(images+adv_noise)

            # Calculate loss
            if self.targeted:
                cost = loss(outputs, target_labels)
            else:
                cost = -loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_noise, retain_graph=False, create_graph=False
            )[0]

            grad = normalize_b(grad, self.eps_for_division, p=self.p)
            
            adv_noise = adv_noise.detach() - self.alpha * grad
            
            clip_noise = norm_h(adv_noise, p=self.p) > self.eps
            adv_noise = torch.where(clip_noise, self.eps * normalize_b(adv_noise, self.eps_for_division, p=self.p), adv_noise)
            
        return adv_noise