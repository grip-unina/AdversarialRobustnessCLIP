'''
from: https://github.com/paarthneekhara/AdversarialDeepFakes
'''

from torch import autograd
import torch
import torch.nn as nn
from . import robust_transforms as rt

def predict_with_model(preprocessed_image, model, model_type, post_function=nn.Softmax(dim=1), resize=False, cuda=True):
    """
    Adapted predict_for_model for attack. Differentiable image pre-processing.
    Predicts the label of an input image. Performs resizing and normalization before feeding in image.

    :param image: torch tenosr (bs, c, h, w)
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real), output probs, logits
    """
    
    # Model prediction

    # differentiable resizing: doing resizing here instead of preprocessing
    if resize == False or model_type == "resnet50":
        resized_image = preprocessed_image
    else:
        resized_image = nn.functional.interpolate(preprocessed_image, size = (224, 224), mode = "bicubic", align_corners = True)
    if cuda:
        resized_image = resized_image.cuda()
    logits = model(resized_image)
    output = post_function(logits)

    # Cast to desired
    _, prediction = torch.max(output, 1)    # argmax
    prediction = float(prediction.cpu().numpy())

    # print ("prediction", prediction)
    # print ("output", output)
    return int(prediction), output, logits


def robust_fgsm(input_img, label, model, model_type, cuda = True, 
    max_iter = 100, alpha = 1/255.0, 
    eps = 16/255.0, desired_acc = 0.95,
    transform_set = {"gauss_noise", "gauss_blur", "translation", "resize"}
    ):

    def _get_transforms(apply_transforms = {"gauss_noise", "gauss_blur", "translation", "resize"}):
        
        transform_list = [
            lambda x: x,
        ]

        if "gauss_noise" in apply_transforms:
            transform_list += [
                lambda x: rt.add_gaussian_noise(x, 0.01, cuda = cuda),
            ]

        if "gauss_blur" in apply_transforms:
            transform_list += [
                lambda x: rt.gaussian_blur(x, kernel_size = (5, 5), sigma=(5., 5.), cuda = cuda),
                lambda x: rt.gaussian_blur(x, kernel_size = (5, 5), sigma=(10., 10.), cuda = cuda),
                lambda x: rt.gaussian_blur(x, kernel_size = (7, 7), sigma=(5., 5.), cuda = cuda),
                lambda x: rt.gaussian_blur(x, kernel_size = (7, 7), sigma=(10., 10.), cuda = cuda),
            ]

        if "translation" in apply_transforms:
            transform_list += [
                lambda x: rt.translate_image(x, 10, 10, cuda = cuda),
                lambda x: rt.translate_image(x, 10, -10, cuda = cuda),
                lambda x: rt.translate_image(x, -10, 10, cuda = cuda),
                lambda x: rt.translate_image(x, -10, -10, cuda = cuda),
                lambda x: rt.translate_image(x, 20, 20, cuda = cuda),
                lambda x: rt.translate_image(x, 20, -20, cuda = cuda),
                lambda x: rt.translate_image(x, -20, 10, cuda = cuda),
                lambda x: rt.translate_image(x, -20, -20, cuda = cuda),
            ]

        if "resize" in apply_transforms:
            transform_list += [
                lambda x: rt.compress_decompress(x, 0.1, cuda = cuda),
                lambda x: rt.compress_decompress(x, 0.2, cuda = cuda),
                lambda x: rt.compress_decompress(x, 0.3, cuda = cuda),
            ]

        return transform_list

    input_var = autograd.Variable(input_img, requires_grad=True)

    #target_var = autograd.Variable(torch.LongTensor([1-label]))
    #if cuda:
    #    target_var = target_var.cuda()
    target = 1 - label

    iter_no = 0
    
    loss_criterion = nn.CrossEntropyLoss()

    while iter_no < max_iter:
        transform_functions = _get_transforms(transform_set)
        
        loss = 0

        all_fooled = True
        #print ("**** Applying Transforms ****")
        for transform_fn in transform_functions:
            
            transformed_img = transform_fn(input_var)
            prediction, output, logits = predict_with_model(transformed_img, model, model_type, cuda=cuda)

            if output[0][target] < desired_acc:
                all_fooled = False

            loss += torch.clamp(logits[0][label]-logits[0][target] + 10, min = 0.0)

            
            # loss += loss_criterion(logits, target_var)

        #print ("*** Finished Transforms **, all fooled", all_fooled)
        if all_fooled:
            break

        loss /= (1. * len(transform_functions))
        if input_var.grad is not None:
            input_var.grad.data.zero_() # just to ensure nothing funny happens
        loss.backward()

        step_adv = input_var.detach() - alpha * torch.sign(input_var.grad.detach())
        total_pert = step_adv - input_img
        total_pert = torch.clamp(total_pert, -eps, eps)
        
        input_adv = input_img + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)
        
        input_var.data = input_adv.detach()

        iter_no += 1

    l_inf_norm = torch.max(torch.abs((input_var - input_img))).item()
    #print ("L infinity norm", l_inf_norm, l_inf_norm * 255.0)
    
    meta_data = {
        'attack_iterations' : iter_no,
        'l_inf_norm' : l_inf_norm,
        'l_inf_norm_255' : round(l_inf_norm * 255.0),
        'noise': (input_var - input_img)
    }

    return input_var, meta_data
