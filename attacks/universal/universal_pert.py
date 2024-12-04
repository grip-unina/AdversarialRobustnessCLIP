'''
from: https://github.com/qilong-zhang/Pytorch_Universal-adversarial-perturbation
'''

import numpy as np
from .deepfool import deepfool
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from torch.autograd import Variable
from networks import load_network

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def universal_perturbation(dataset: Dataset,
                           valset: Dataset,
                           namenet: str,
                           delta=0.01,
                           max_iter_uni = np.inf,
                           eps=10,
                           p=np.inf,
                           num_classes=2,
                           overshoot=0.02,
                           max_iter_df=10,
                           out_dir='./perturbations/',
                           device='cuda'):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param net_dict: network dict.
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :param device: cuda device (default = cuda)
    :return: the universal perturbation.
    """
    xi = eps / 255.0
    #f: feedforward function (input: images, output: values of activation BEFORE softmax)
    f = load_network(namenet)
    #print('Net:', namenet)
    print('p =', p, 'eps =', xi)

    v = 0
    fooling_rate = 0.0
    best_fooling = 0.0
    num_images = len(valset) # The length of testing data
    iter_uni = 0

    while fooling_rate < 1-delta and iter_uni < max_iter_uni:
        iter_uni+=1
        # Shuffle the dataset
        data_loader = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory=True)

        # Go through the data set and compute the perturbation increments sequentially
        k = 0
        f.to(device).eval()
        for cur_img in tqdm(data_loader):
            k += 1
            cur_img = cur_img.to(device)
            per = Variable(cur_img + torch.tensor(v).to(device), requires_grad = True)
            if int(f(cur_img).argmax()) == int(f(per).argmax()):
                # Compute adversarial perturbation
                f.zero_grad()
                dr, iter = deepfool(per,
                                   f,
                                   num_classes = num_classes,
                                   overshoot = overshoot,
                                   max_iter = max_iter_df)
                # print('dr = ', abs(dr).max())

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    v = proj_lp(v, xi, p)

        # Perturb the dataset with computed perturbation
        # dataset_perturbed = dataset + v
        est_labels_orig = torch.zeros((num_images)).to(device)
        est_labels_pert = torch.zeros((num_images)).to(device)

        batch_size = 50
        val_loader = DataLoader(
            valset,
            batch_size = batch_size, shuffle=False,
            num_workers = 8, pin_memory=True)

        # Compute the estimated labels in batches
        ii = 0
        with torch.no_grad():
            for img_batch in tqdm(val_loader):
                m = (ii * batch_size)
                M = min((ii + 1) * batch_size, num_images)
                img_batch = img_batch.to(device)
                per_img_batch = (img_batch + torch.tensor(v).to(device)).to(device)
                ii += 1
                # print(img_batch.shape)
                # print(m, M)
                est_labels_orig[m:M] = torch.argmax(f(img_batch), dim=1)
                est_labels_pert[m:M] = torch.argmax(f(per_img_batch), dim=1)

            # Compute the fooling rate
            fooling_rate = torch.sum(est_labels_pert != est_labels_orig).float() / num_images
            print(torch.sum(est_labels_pert != est_labels_orig).float())
            print('FOOLING RATE = ', fooling_rate)
            if fooling_rate > best_fooling:
                best_fooling = fooling_rate
                #Reset iter_uni
                iter_uni = 0
            print('Best Fooling Rate = ', best_fooling)
            out_dir.mkdir(parents=True, exist_ok=True)
            pertbation_path = out_dir / f'{fooling_rate*100:.2f}.npy'

            np.save(pertbation_path, v.squeeze().transpose(1, 2, 0))

    return v