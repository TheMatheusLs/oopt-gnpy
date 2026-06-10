#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.core.numba_optimizations: Numba-accelerated implementations of CPU-intensive functions
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors

"""
gnpy.core.numba_optimizations
==============================

Numba-accelerated implementations for performance-critical functions.

This module provides JIT-compiled versions of computationally intensive functions
from science_utils.py. When Numba is available, these implementations can provide 
significant speedups (10-100x) for CPU-bound operations.

If Numba is not installed, the module gracefully falls back to the original 
implementations without affecting functionality.
"""

try:
    from numba import jit, prange #type: ignore
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

import numpy as np
import math
import cmath


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def is_numba_available():
    return NUMBA_AVAILABLE


# ============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# ============================================================================
@jit(nopython=True, cache=True)
def scalar_raised_cosine(base_freq, pass_band, stop_band, ts, roll_off):
    """Cálculo escalar inline para evitar overhead de alocação."""
    abs_freq = abs(base_freq)
    if abs_freq <= pass_band:
        return 1.0
    elif abs_freq < stop_band:
        return 0.5 * (1.0 + math.cos(math.pi * ts / roll_off * (abs_freq - pass_band)))
    return 0.0


@jit(nopython=True, parallel=True, cache=True)
def raised_cosine_numba(frequency, channel_frequency, channel_baud_rate, channel_roll_off):
    """API Vetorizada para ser consumida externamente (ex: cálculo de rc1)."""
    n = frequency.size
    out = np.zeros(n)
    ts = 1.0 / channel_baud_rate
    pass_band = (1.0 - channel_roll_off) * channel_baud_rate / 2.0
    stop_band = (1.0 + channel_roll_off) * channel_baud_rate / 2.0
    
    for i in prange(n):
        out[i] = scalar_raised_cosine(frequency[i] - channel_frequency, pass_band, stop_band, ts, channel_roll_off)
    return out


@jit(nopython=True, cache=True, parallel=True)
def _approx_psi_leff_computation_numba(loss_profile, delta_z, z_size):
    nfreq = loss_profile.shape[0]
    n_segments = z_size - 1
    leff2 = np.zeros(nfreq)
    
    for i in prange(nfreq):
        sum_leff = 0.0
        for j in range(n_segments):
            loss_lin_j0 = math.log(loss_profile[i, j])
            loss_lin_j1 = math.log(loss_profile[i, j + 1])
            alpha_val = (loss_lin_j1 - loss_lin_j0) / delta_z[j]
            
            abs_alpha = abs(alpha_val)
            if abs_alpha > 1e-15:
                loss_diff = loss_profile[i, j + 1] - loss_profile[i, j]
                sum_leff += abs(loss_diff / math.sqrt(abs_alpha)) * (alpha_val / abs_alpha)
                
        leff2[i] = sum_leff ** 2
        
    return leff2

@jit(nopython=True, cache=True)
def _generalized_psi_inner_loop_numba(f1_array, f2_array, rc1, f_eval, 
                                     cut_frequency, cut_baud_rate, cut_roll_off,
                                     pump_frequency, pump_baud_rate, pump_roll_off,
                                     beta2, beta3, f_ref_beta, rho_pump, z, alpha):
    
    integrand_f1 = np.zeros(f1_array.size)
    f2_size = f2_array.size
    
    ts_cut = 1.0 / cut_baud_rate
    pass_band_cut = (1.0 - cut_roll_off) * cut_baud_rate / 2.0
    stop_band_cut = (1.0 + cut_roll_off) * cut_baud_rate / 2.0
    
    ts_pump = 1.0 / pump_baud_rate
    pass_band_pump = (1.0 - pump_roll_off) * pump_baud_rate / 2.0
    stop_band_pump = (1.0 + pump_roll_off) * pump_baud_rate / 2.0
    
    integrand_f2 = np.zeros(f2_size)
    rc2 = np.zeros(f2_size)
    
    for j in range(f2_size):
        rc2[j] = scalar_raised_cosine(f2_array[j] - cut_frequency, pass_band_cut, stop_band_cut, ts_cut, cut_roll_off)
        
    z_len = len(z)

    for i in range(f1_array.size):
        f1 = f1_array[i]
        for j in range(f2_size):
            f2 = f2_array[j]
            f3 = f1 + f2 - f_eval
            
            rc3_val = scalar_raised_cosine(f3 - pump_frequency, pass_band_pump, stop_band_pump, ts_pump, pump_roll_off)
            
            if rc3_val == 0.0 or rc2[j] == 0.0 or rc1[i] == 0.0:
                integrand_f2[j] = 0.0
                continue
                
            delta_beta = 4.0 * math.pi**2 * (f1 - f_eval) * (f2 - f_eval) * \
                         (beta2 + math.pi * beta3 * (f1 + f2 - 2.0 * f_ref_beta))
            
            w = complex(-alpha, delta_beta)
            rho_nli = 0.0
            
            if abs(w) > 1e-15:
                generalized_rho = (rho_pump[-1]**2 * cmath.exp(w * z[-1]) - rho_pump[0]**2 * cmath.exp(w * z[0])) / w
                for z_ind in range(z_len - 1):
                    dz = z[z_ind + 1] - z[z_ind]
                    derivative_rho = (rho_pump[z_ind + 1]**2 - rho_pump[z_ind]**2) / dz
                    generalized_rho -= derivative_rho * (cmath.exp(w * z[z_ind + 1]) - cmath.exp(w * z[z_ind])) / (w**2)
                    
                rho_nli = generalized_rho.real**2 + generalized_rho.imag**2
            
            integrand_f2[j] = rc1[i] * rc2[j] * rc3_val * rho_nli
            
        integral = 0.0
        for j in range(f2_size - 1):
            dx = f2_array[j+1] - f2_array[j]
            integral += 0.5 * (integrand_f2[j] + integrand_f2[j+1]) * dx
            
        integrand_f1[i] = integral
        
    return integrand_f1