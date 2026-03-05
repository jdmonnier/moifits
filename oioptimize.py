'''
This code is adapted from the Julia OITOOLS package written by Fabien Baron, freely available on GitHub.
'''

import torch
import numpy as np
from .oichi2 import image_to_vis, vis_to_v2, vis_to_t3, mod360, nfft_adjoint
from .vis_functions import visibility_ud, dvisibility_ud
from scipy.special import j0, j1

def chi2_sparco_f(x, params, ftplan, data, verbose=True, weights=[1.0, 1.0, 1.0]):
    """
    Criterion function for NFFT with chromatic model (SPARCO).
    
    The chromatism is defined as follows:
             f_star_0 * (lambda/lambda_0)^-4 * V_star + (1-f_star_0-f_bg_0)*(lambda/lambda_0)^d_ind * V_env
    V_tot = ------------------------------------------------------------------------------------------------
                    (f_star_0 + f_bg_0) * (lambda/lambda_0)^-4 + (1-f_star_0-f_bg_0)*(lambda/lambda_0)^d_ind
    
    Args:
        x: 2D image array (nx, nx) - the circumstellar environment
        params: Parameter array [f_star_0, f_bg_0, diameter, d_ind, lambda_0]
            params[0] = f_star_0: stellar flux fraction at lambda_0
            params[1] = f_bg_0: background flux fraction at lambda_0
            params[2] = diameter: stellar diameter in mas
            params[3] = d_ind: environment power law index (fixed)
            params[4] = lambda_0: reference wavelength (fixed), e.g., 1.65e-6
        ftplan: List of NFFTPlan objects [uv, vis, v2, t3_1, t3_2, t3_3]
        data: OIData object
        verbose: Print chi2 components
        weights: [w_v2, w_t3amp, w_t3phi]
        
    Returns:
        chi2: Total chi-squared value
    """
    # Extract parameters
    lambda_0 = params[4]
    lambda_obs = data.uv_lam
    
    # Compute chromatic terms
    alpha = (lambda_obs / lambda_0)**(-4.0)
    beta = (lambda_obs / lambda_0)**(params[3] - 4.0)
    
    flux_star = params[0] * alpha
    flux_bg = params[1] * alpha
    flux_env = (1.0 - params[0] - params[1]) * beta
    
    # Compute visibilities
    V_star = visibility_ud([params[2]], data.uv)
    V_env = image_to_vis(x, ftplan[0])
    
    # Combined visibility
    cvis_model = (flux_star * V_star + flux_env * V_env) / (flux_star + flux_env + flux_bg)
    
    # Compute observables
    v2_model = vis_to_v2(cvis_model, data.indx_v2)
    t3_model, t3amp_model, t3phi_model = vis_to_t3(cvis_model,
                                                     data.indx_t3_1,
                                                     data.indx_t3_2,
                                                     data.indx_t3_3)
    
    # Compute chi-squared for each observable
    chi2_v2 = chi2_t3amp = chi2_t3phi = 0.0
    
    if weights[0] > 0:
        chi2_v2 = np.sum(((v2_model - data.v2) / data.v2_err)**2)
    
    if weights[1] > 0:
        chi2_t3amp = np.sum(((t3amp_model - data.t3amp) / data.t3amp_err)**2)
    
    if weights[2] > 0:
        chi2_t3phi = np.sum((mod360(t3phi_model - data.t3phi) / data.t3phi_err)**2)
    
    if verbose:
        print(f"\033[31mV2: {chi2_v2/data.nv2:.4f}\033[0m ", end='')
        print(f"\033[34mT3A: {chi2_t3amp/data.nt3amp:.4f}\033[0m ", end='')
        print(f"\033[32mT3P: {chi2_t3phi/data.nt3phi:.4f}\033[0m")
    
    return weights[0]*chi2_v2 + weights[1]*chi2_t3amp + weights[2]*chi2_t3phi

def optimize_sparco_parameters(params_start, x, ftplan, data, 
                                weights=[1.0, 1.0, 1.0],
                                lb=[0.0, 0.0, 0.0, -20.0],
                                ub=[1.0, 1.0, 1.0, 20.0],
                                max_iter=100,
                                verbose=True):
    """
    Optimize chromatic model parameters using PyTorch LBFGS.
    
    Optimizes the first 4 parameters [f_star_0, f_bg_0, diameter, d_ind]
    while keeping the reference wavelength (params[4]) fixed.
    
    Args:
        params_start: Initial parameter array [f_star_0, f_bg_0, diameter, d_ind, lambda_0]
        x: 2D image array (nx, nx) - the circumstellar environment
        ftplan: List of NFFTPlan objects
        data: OIData object
        weights: [w_v2, w_t3amp, w_t3phi]
        lb: Lower bounds for the 4 optimized parameters
        ub: Upper bounds for the 4 optimized parameters
        max_iter: Maximum number of iterations
        verbose: Print optimization progress
        
    Returns:
        min_chi2: Minimum chi-squared value achieved
        params_opt: Optimized parameters [f_star_0, f_bg_0, diameter, d_ind, lambda_0]
        result: Optimization result info
    """   
    # Extract the 4 parameters to optimize (keep lambda_0 fixed)
    params_to_opt = np.array(params_start[:4], dtype=np.float64)
    lambda_0_fixed = params_start[4]
    
    # Convert to PyTorch tensor with gradient tracking
    params_torch = torch.tensor(params_to_opt, dtype=torch.float64, requires_grad=True)
    
    print("Starting SPARCO parameter optimization...")
    # LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        [params_torch],
        max_iter=20,  # max iterations per step
        line_search_fn='strong_wolfe'
    )
    
    print("Optimizing parameters...")
    # Optimization history
    history = {'chi2': [], 'params': []}
    iteration = [0]  # Use list for closure scope
    
    def closure():
        optimizer.zero_grad()
        
        # Get current parameter values and apply bounds via clamping
        params_clamped = torch.clamp(params_torch, 
                                     torch.tensor(lb, dtype=torch.float64),
                                     torch.tensor(ub, dtype=torch.float64))
        
        # Build full parameter array with fixed lambda_0
        params_current = np.concatenate([params_clamped.detach().numpy(), [lambda_0_fixed]])
        
        # Build combined vector [params | image] and evaluate joint objective/gradient
        img_flat = x.flatten(order='F')
        x_combined = np.concatenate([params_current, img_flat])
        g_combined = np.zeros_like(x_combined)
        chi2_val = chi2_sparco_fg(
            x_combined, g_combined, ftplan, data,
            nparams=len(params_current), verbose=False, weights=weights
        )
        
        chi2_torch = torch.tensor(chi2_val, dtype=torch.float64, requires_grad=True)
        
        # Gradient: first four entries correspond to optimized parameters
        grad = g_combined[:4]
        params_torch.grad = torch.tensor(grad, dtype=torch.float64)
        
        # Store history
        history['chi2'].append(chi2_val)
        history['params'].append(params_current.copy())
        
        if verbose and iteration[0] % 10 == 0:
            norm_chi2 = chi2_val / max(data.nv2 + data.nt3phi, 1)
            print(f"Iter {iteration[0]:3d}: chi2 = {chi2_val:.6e} "
                  f"(norm={norm_chi2:.6e}), params = "
                  f"[{params_current[0]:.3f}, {params_current[1]:.3f}, "
                  f"{params_current[2]:.3f}, {params_current[3]:.3f}]")
        
        iteration[0] += 1
        
        return chi2_torch
    
    # Run optimization
    try:
        for i in range(max_iter // 20):  # Outer loop
            optimizer.step(closure)
            
            # Check convergence
            if len(history['chi2']) > 2:
                if abs(history['chi2'][-1] - history['chi2'][-2]) < 1e-6:
                    if verbose:
                        print("Converged!")
                    break
    except Exception as e:
        print(f"Optimization stopped: {e}")
    
    # Get final results
    params_final = torch.clamp(params_torch, 
                               torch.tensor(lb, dtype=torch.float64),
                               torch.tensor(ub, dtype=torch.float64))
    params_opt = np.concatenate([params_final.detach().numpy(), [lambda_0_fixed]])
    min_chi2 = history['chi2'][-1] if history['chi2'] else float('inf')
    
    result = {
        'success': len(history['chi2']) > 0,
        'nit': len(history['chi2']),
        'history': history
    }
    
    if verbose:
        norm_chi2 = min_chi2 / max(data.nv2 + data.nt3phi, 1)
        print(f"\nOptimization complete:")
        print(f"  Final chi2: {min_chi2:.6e} (norm={norm_chi2:.6e})")
        print(f"  Final params: {params_opt}")
        print(f"  Iterations: {result['nit']}")
    
    return min_chi2, params_opt, result

def chi2_sparco_fg(x_combined, g_combined, ftplan, data, nparams, verbose=True, weights=[1.0, 1.0, 1.0]):
    """
    Criterion and gradient for NFFT with chromatic model (SPARCO).
    Computes chi2 and gradients w.r.t. parameters and image, matching the Julia implementation.

    The chromatism is defined as follows:
             f_star_0 * (lambda/lambda_0)^-4 * V_star + (1-f_star_0-f_bg_0)*(lambda/lambda_0)^d_ind * V_env
    V_tot = ------------------------------------------------------------------------------------------------
                    (f_star_0 + f_bg_0) * (lambda/lambda_0)^-4 + (1-f_star_0-f_bg_0)*(lambda/lambda_0)^d_ind

    Args:
        x_combined: 1D array concatenating parameters and the flattened image vector.
                    params = [f_star_0, f_bg_0, diameter, d_ind, lambda_0]
        g_combined: 1D array to be filled with gradients.
        ftplan: List of NFFTPlan objects [uv, vis, v2, t3_1, t3_2, t3_3]
        data: OIData object
        nparams: Number of parameters at the start of x_combined.
        verbose: Print chi2 components
        weights: [w_v2, w_t3amp, w_t3phi]

    Returns:
        chi2: Total chi-squared value
    """
    # 1. Unpack parameters and image
    params = x_combined[:nparams]
    nx = ftplan[0].nx
    img_vec = x_combined[nparams:]
    x = img_vec.reshape((nx, nx))

    # 2. Forward model calculation
    lambda_0 = params[4]
    lambda_obs = data.uv_lam
    
    alpha = (lambda_obs / lambda_0)**(-4.0)
    beta = (lambda_obs / lambda_0)**(params[3] - 4.0)
    
    flux_star = params[0] * alpha
    flux_bg = params[1] * alpha
    flux_env = (1.0 - params[0] - params[1]) * beta
    
    V_star = visibility_ud([params[2]], data.uv)
    V_env = image_to_vis(x, ftplan[0])
    
    u = flux_star * V_star + flux_env * V_env
    v = flux_star + flux_env + flux_bg
    cvis_model = u / v
    
    v2_model = vis_to_v2(cvis_model, data.indx_v2)
    t3_model, t3amp_model, t3phi_model = vis_to_t3(cvis_model,
                                                     data.indx_t3_1,
                                                     data.indx_t3_2,
                                                     data.indx_t3_3)
    
    # 3. Compute chi-squared for each observable
    chi2_v2 = np.sum(((v2_model - data.v2) / data.v2_err)**2) if data.nv2 > 0 else 0.0
    chi2_t3amp = np.sum(((t3amp_model - data.t3amp) / data.t3amp_err)**2) if data.nt3amp > 0 else 0.0
    chi2_t3phi = np.sum((mod360(t3phi_model - data.t3phi) / data.t3phi_err)**2) if data.nt3phi > 0 else 0.0

    if verbose:
        print(f"\033[31mV2: {chi2_v2/data.nv2:.4f}\033[0m ", end='')
        print(f"\033[34mT3A: {chi2_t3amp/data.nt3amp:.4f}\033[0m ", end='')
        print(f"\032mT3P: {chi2_t3phi/data.nt3phi:.4f}\033[0m")

    # 4. Compute gradients
    g_combined.fill(0.0)

    # Helper function to compute gradient contribution from a visibility derivative
    def delta_chi2(dcvis_model):
        dchi2 = 0.0
        # V2 contribution
        if weights[0] > 0 and data.nv2 > 0:
            dchi2 += 4.0 * np.sum(((v2_model - data.v2) / data.v2_err**2) * 
                                  np.real(np.conj(cvis_model[data.indx_v2]) * dcvis_model[data.indx_v2]))
        
        # T3amp contribution
        if weights[1] > 0 and data.nt3amp > 0:
             dchi2 += 2.0 * np.sum(((t3amp_model - data.t3amp) / data.t3amp_err**2) * (
                np.real(cvis_model[data.indx_t3_1] * np.conj(dcvis_model[data.indx_t3_1])) / np.abs(cvis_model[data.indx_t3_1]) * np.abs(cvis_model[data.indx_t3_2]) * np.abs(cvis_model[data.indx_t3_3]) +
                np.real(cvis_model[data.indx_t3_2] * np.conj(dcvis_model[data.indx_t3_2])) / np.abs(cvis_model[data.indx_t3_2]) * np.abs(cvis_model[data.indx_t3_1]) * np.abs(cvis_model[data.indx_t3_3]) +
                np.real(cvis_model[data.indx_t3_3] * np.conj(dcvis_model[data.indx_t3_3])) / np.abs(cvis_model[data.indx_t3_3]) * np.abs(cvis_model[data.indx_t3_1]) * np.abs(cvis_model[data.indx_t3_2])
            ))

        # T3phi contribution
        if weights[2] > 0 and data.nt3phi > 0:
            t3 = cvis_model[data.indx_t3_1] * cvis_model[data.indx_t3_2] * cvis_model[data.indx_t3_3]
            dt3 = (dcvis_model[data.indx_t3_1] * cvis_model[data.indx_t3_2] * cvis_model[data.indx_t3_3] + 
                   cvis_model[data.indx_t3_1] * dcvis_model[data.indx_t3_2] * cvis_model[data.indx_t3_3] +
                   cvis_model[data.indx_t3_1] * cvis_model[data.indx_t3_2] * dcvis_model[data.indx_t3_3])
            dchi2 += -360.0 / np.pi * np.sum((mod360(t3phi_model - data.t3phi) / data.t3phi_err**2) *
                                            np.imag(np.conj(t3) * dt3) / np.abs(t3)**2)
        #print(f"Delta chi2 contribution: {dchi2:.6e}")
        return dchi2

    # 4a. Parameter gradients
    # Gradient w.r.t. f_star_0 (params[0])
    du_dfs = alpha * V_star - beta * V_env
    dv_dfs = alpha - beta
    dcvis_dfs0 = (du_dfs * v - u * dv_dfs) / (v**2)
    g_combined[0] = delta_chi2(dcvis_dfs0)
    
    # Gradient w.r.t. f_bg_0 (params[1])
    du_dfg = -beta * V_env
    dv_dfg = alpha - beta
    dcvis_dfg0 = (du_dfg * v - u * dv_dfg) / (v**2)
    g_combined[1] = 0.0
    
    # Gradient w.r.t. diameter (params[2])
    dVstar_dD = dvisibility_ud([params[2]], data.uv)
    du_dD = flux_star * dVstar_dD
    dcvis_dD = du_dD / v
    #print(f"real: {np.min(np.real(dcvis_dD))} → {np.max(np.real(dcvis_dD))}")
    #print(f"imag: {np.min(np.imag(dcvis_dD))} → {np.max(np.imag(dcvis_dD))}")
    g_combined[2] = delta_chi2(dcvis_dD)
    
    # Gradient w.r.t. spectral index (params[3])
    log_ratio = np.log(lambda_obs / lambda_0)
    du_dindx = log_ratio * flux_env * V_env
    dv_dindx = log_ratio * flux_env
    dcvis_dindx = (du_dindx * v - u * dv_dindx) / (v**2)
    g_combined[3] = delta_chi2(dcvis_dindx)
    
    # Gradient w.r.t. lambda_0 (params[4]) is 0 as it's fixed.
    g_combined[4] = 0.0

    #print(f"Parameter gradients: {g_combined[:nparams]}")
    # 4b. Image gradients
    imratio = flux_env / v
    g_v2 = g_t3amp = g_t3phi = np.zeros((nx, nx), dtype=np.float64)

    if weights[0] > 0 and data.nv2 > 0:
        g_v2 = np.real(nfft_adjoint(ftplan[2], 4.0 * ((v2_model - data.v2) / data.v2_err**2) * 
                            cvis_model[data.indx_v2] * imratio[data.indx_v2], real_output=False))
        #print(f"||g_v2|| = {np.linalg.norm(g_v2):.6e}")
    if weights[1] > 0 and data.nt3amp > 0:
        dT3amp_term = (t3amp_model - data.t3amp) / (data.t3amp_err ** 2)
        g1 = (
            dT3amp_term
            * cvis_model[data.indx_t3_1]
            * imratio[data.indx_t3_1]
            * np.abs(cvis_model[data.indx_t3_2])
            * np.abs(cvis_model[data.indx_t3_3])
            / np.abs(cvis_model[data.indx_t3_1])
        )

        g2 = (
            dT3amp_term
            * cvis_model[data.indx_t3_2]
            * imratio[data.indx_t3_2]
            * np.abs(cvis_model[data.indx_t3_1])
            * np.abs(cvis_model[data.indx_t3_3])
            / np.abs(cvis_model[data.indx_t3_2])
        )

        g3 = (
            dT3amp_term
            * cvis_model[data.indx_t3_3]
            * imratio[data.indx_t3_3]
            * np.abs(cvis_model[data.indx_t3_1])
            * np.abs(cvis_model[data.indx_t3_2])
            / np.abs(cvis_model[data.indx_t3_3])
        )
        g_t3amp = np.real(
            nfft_adjoint(ftplan[3], g1, real_output=False) +
            nfft_adjoint(ftplan[4], g2, real_output=False) +
            nfft_adjoint(ftplan[5], g3, real_output=False)
        )
        #print(f"||g_t3amp|| = {np.linalg.norm(g_t3amp):.6e}")
        
    if weights[2] > 0 and data.nt3phi > 0:
        phase_weight = (mod360(t3phi_model - data.t3phi) / data.t3phi_err**2) / np.abs(t3_model)**2

        g1 = phase_weight * imratio[data.indx_t3_1] * np.conj(cvis_model[data.indx_t3_2] * cvis_model[data.indx_t3_3]) * t3_model
        g2 = phase_weight * imratio[data.indx_t3_2] * np.conj(cvis_model[data.indx_t3_1] * cvis_model[data.indx_t3_3]) * t3_model
        g3 = phase_weight * imratio[data.indx_t3_3] * np.conj(cvis_model[data.indx_t3_1] * cvis_model[data.indx_t3_2]) * t3_model

        g_t3phi = -360.0 / np.pi * np.imag(
            nfft_adjoint(ftplan[3], g1, real_output=False) +
            nfft_adjoint(ftplan[4], g2, real_output=False) +
            nfft_adjoint(ftplan[5], g3, real_output=False)
        )
        #print(f"||g_t3phi|| = {np.linalg.norm(g_t3phi):.6e}")
    else:
        print("No T3phi data; skipping gradient computation for T3phi.")

    g_img = weights[0] * g_v2 + weights[1] * g_t3amp + weights[2] * g_t3phi
    g_combined[nparams:] = g_img.flatten()

    return weights[0]*chi2_v2 + weights[1]*chi2_t3amp + weights[2]*chi2_t3phi
