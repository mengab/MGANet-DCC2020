import torch
import torch.nn as nn


def EPE(input_image, target_image,L_model=None):

    loss_L2 = L_model(input_image,target_image)

    return loss_L2
    # EPE_map = torch.norm(target_image-input_image,2,1)
    # batch_size = EPE_map.size(0)
    #
    # if mean:
    #     return EPE_map.mean()
    # else:
    #     return EPE_map.sum()/batch_size


def sparse_max_pool(input, size):
    positive = (input > 0).float()
    negative = (input < 0).float()
    output = nn.functional.adaptive_max_pool2d(input * positive, size) - nn.functional.adaptive_max_pool2d(-input * negative, size)
    return output


def multiscaleEPE(network_output, target_image, weights=None, L_model=None):
    def one_scale(output, target, L_model):

        b, _, h, w = output.size()

        target_scaled = nn.functional.adaptive_avg_pool2d(target, (h, w))

        return EPE(output, target_scaled, L_model)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [ 1.0/32,1.0/16.0, 1.0/8.0, 1.0/4.0, 1.0/2.0]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_image,L_model)
    return loss


