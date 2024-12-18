################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), (
        "The reparameterization trick got a negative std as input. "
        + "Are you sure your input is std and not log_std?"
    )
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    eps = torch.randn_like(std)
    # reparameterize
    z = mean + std * eps
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # compute components
    var = torch.exp(2 * log_std)
    kld = 0.5 * (var + mean**2 - 1 - 2 * log_std)
    # sum over last dimension
    KLD = torch.sum(kld, dim=-1)
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # excluding batch
    _, C, H, W = img_shape
    num_dims = C * H * W

    # convert from nats to bits and normalize by number of dimensions
    bpd = (
        elbo * (1.0 / num_dims) * (1.0 / torch.log(torch.tensor(2.0)))
    )  # log2(e) = 1/log(2)
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # normal(0,1) distribution to access icdf
    normal = torch.distributions.Normal(0, 1)

    # generate the percentiles
    percentiles = torch.linspace(
        0.5 / grid_size, (grid_size - 0.5) / grid_size, grid_size
    )

    # convert percentiles to latent values using the inverse CDF
    z_values = normal.icdf(percentiles)

    # create a meshgrid of Z values (z_x, z_y)
    z_x, z_y = torch.meshgrid(z_values, z_values)
    z_x = z_x.flatten()
    z_y = z_y.flatten()

    # create the latent codes [grid_size^2, 2]
    Z = torch.stack([z_x, z_y], dim=-1).to(next(decoder.parameters()).device)

    # decode the latent codes
    decoded = decoder(Z)  # shape: [grid_size^2, C, H, W] (model dependent)

    # apply softmax over the channel dimension if the decoder outputs logits for a categorical distribution
    # if the model is Bernoulli with logits, you might use torch.sigmoid instead.
    # adjust based on your model.
    decoded = torch.nn.functional.softmax(decoded, dim=1)

    # Convert 16-channel to single-channel by taking a weighted sum and then divide by 15
    weights = torch.arange(16, dtype=torch.float32, device=decoded.device).view(
        1, -1, 1, 1
    )
    decoded = (
        torch.sum(decoded * weights, dim=1, keepdim=True) / 15.0
    )  # now shape: [grid_size^2, 1, H, W]
    # # Convert 16-channel to single-channel by taking argmax
    # decoded = decoded.argmax(
    #     dim=1, keepdim=True
    # ).float()  # now shape: [grid_size^2, 1, H, W]

    # Make a grid of images
    img_grid = make_grid(decoded, nrow=grid_size, normalize=False, padding=1)
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid
