import drjit as dr
import torch
from math import log

# Based on https://github.com/rubenwiersma/svbrdf_uncertainty/blob/main/svbrdf_uncertainty/util/color.py


def linear_to_gamma(rgb_linear, threshold=0.0031308, a=0.055, gamma=2.4):
    """Converts linear RGB to gamma corrected RGB.

    Args:
        rgb_linear (torch.Tensor): Linear RGB values in [0, 1].
        threshold (float): Threshold for gamma correction. Uses a linear approximation below this value.
        a (float): Constant for gamma correction.
        gamma (float): Gamma correction exponent.
    """
    return torch.where(
        rgb_linear > threshold,
        (1 + a) * (rgb_linear ** (1.0 / gamma)) - a,
        12.92 * rgb_linear,
    )


def linear_to_gamma_dr(rgb_linear):
    return 1.055 * (dr.power(rgb_linear, (1.0 / 2.4))) - 0.055


def gamma_to_linear(rgb_gamma, threshold=0.0031808, a=0.055, gamma=2.4):
    """Converts gamma corrected RGB to linear RGB.

    Args:
        rgb_gamma (torch.Tensor): sRGB values.
        threshold (float): Threshold for gamma correction. Uses a linear approximation below this value.
        a (float): Constant for gamma correction.
        gamma (float): Gamma correction exponent.
    """
    return torch.where(
        rgb_gamma > 12.92 * threshold,
        ((rgb_gamma + a) / (1 + a)) ** gamma,
        rgb_gamma / 12.92,
    )


def to_log(luminance, epsilon=1e-6):
    """Converts luminance to log space.

    Args:
        luminance (torch Tensor): Luminance values.
        epsilon (float): Small value to avoid log(0).
    """
    log_epsilon = log(epsilon)
    return (torch.log(luminance + epsilon) - log_epsilon) / (
        log(1.0 + epsilon) - log_epsilon
    )


def to_log_dr(luminance, epsilon=1e-6):
    """Converts luminance to log space.

    Args:
        luminance (drjit Tensor): Luminance values.
        epsilon (float): Small value to avoid log(0).
    """
    return (dr.log(luminance + epsilon) - dr.log(epsilon)) / (
        dr.log(1.0 + epsilon) - dr.log(epsilon)
    )
