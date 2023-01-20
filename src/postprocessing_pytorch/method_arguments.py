#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Written by H. Turbé, G. Mengaldo May 2022.
     Arguments for the different interpretability methods
"""

from captum.attr import (
    LRP,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    KernelShap,
    Lime,
    NoiseTunnel,
    Saliency,
    ShapleyValueSampling,
)


bool_multiply_inputs = True
baseline_type = "random"
dict_method_arguments = {
    "integrated_gradients": {
        "captum_method": IntegratedGradients,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        "noback_cudnn": True,
        "batch_size": 8,
    },
    "deeplift": {
        "captum_method": DeepLift,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        "noback_cudnn": True,
        "batch_size": 8,
    },
    "deepliftshap": {
        "captum_method": DeepLiftShap,
        "require_baseline": True,
        "baseline_type": "sample",
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        "noback_cudnn": True,
        "batch_size": 2,
    },
    "gradshap": {
        "captum_method": GradientShap,
        "require_baseline": True,
        "baseline_type": "sample",
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        "noback_cudnn": True,
        "batch_size": 8,
    },
    "shapleyvalue": {
        "captum_method": ShapleyValueSampling,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "kwargs_attribution": {"perturbations_per_eval": 16},
        "noback_cudnn": False,
        "batch_size": 16,
    },
    "kernelshap": {
        "captum_method": KernelShap,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "noback_cudnn": False,
        "batch_size": 8,
    },
    "lime": {
        "captum_method": Lime,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "noback_cudnn": False,
        "batch_size": 8,
    },
    "saliency": {
        "captum_method": Saliency,
        "require_baseline": False,
        "noback_cudnn": True,
        "batch_size": 8,
    },
    "lrp": {
        "captum_method": LRP,
        "require_baseline": False,
        "noback_cudnn": False,
        "batch_size": 8,
    },
}
