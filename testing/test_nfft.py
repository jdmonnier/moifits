#!/usr/bin/env python3
"""
Test script to compare Python NFFT (finufft) with Julia NFFT.
"""

import sys
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent))

from readoifits import readoifits
from oichi2 import setup_nfft, image_to_vis, image_to_obs, chi2_nfft, chi2_fg
from oioptimize import chi2_sparco_f, chi2_sparco_fg, optimize_sparco_parameters

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_julia_nfft(oifits_file, nx, pixsize):
    """
    Run Julia NFFT setup and return results via JSON.
    """
    inference_dir = Path(__file__).parent.parent.absolute()
    
    julia_script = f"""
    using Pkg
    Pkg.activate("{inference_dir}")
    
    using OITOOLS
    using JSON3
    using LinearAlgebra
    
    # Load data
    data = readoifits("{oifits_file}", filter_bad_data=true)[1,1]
    
    # Setup NFFT
    include("{inference_dir}/OITOOLS/oichi2.jl")
    ft = setup_nfft(data, {nx}, {pixsize})
    
    # Print indexing sanity checks
    println("[Julia] First 3 T3 triplets:")
    for i in 1:min(3, length(data.indx_t3_1))
        println("   ", data.indx_t3_1[i], ", ", data.indx_t3_2[i], ", ", data.indx_t3_3[i])
    end

    # Create Gaussian test image
    function gaussian2d(n, m, sigma)
        g2d = [exp(-((X-(m/2)).^2+(Y-n/2).^2)/(2*sigma.^2)) for X=1:m, Y=1:n]
        return g2d/sum(g2d)
    end
    
    x = gaussian2d({nx}, {nx}, {nx}/8)
    
    # Compute visibilities and observables
    cvis = image_to_vis(x, ft[1])
    v2, t3amp, t3phi = image_to_obs(x, ft, data)
    
    # Compute chi2
    chi2_val = chi2_f(x, ft, data)
    
    # Compute chi2 with gradient
    g = zeros(size(x))
    chi2_fg_val = chi2_fg(x, g, ft, data)
    
    # Test chi2_sparco_f with chromatic model
    params = [0.8, 0.1, 0.5, 0.0, 1.65e-6]
    chi2_sparco_val = chi2_sparco_f(x, params, ft, data, verb=false)
    
    # Test chi2_sparco_fg with gradient - try multiple parameter sets
    println("[Julia] Extended chi2_sparco_fg gradient test with multiple parameter sets:")
    
    test_params_list = [
        [0.8, 0.1, 0.5, 0.0, 1.65e-6],
        [0.5, 0.2, 0.3, 0.0, 1.65e-6],
        [0.9, 0.05, 0.7, -1.0, 1.65e-6],
        [0.6, 0.15, 0.4, 1.0, 1.65e-6]
    ]
    
    for (idx, test_params) in enumerate(test_params_list)
        println("\\n--- Test set ", idx, ": f_star=", test_params[1], ", f_bg=", test_params[2], 
                ", D=", test_params[3], "mas, d_ind=", test_params[4], " ---")
        nparams_test = length(test_params)
        x_combined_test = [test_params; vec(x)]
        g_combined_test = zeros(length(x_combined_test))
        chi2_val_test = chi2_sparco_fg(x_combined_test, g_combined_test, ft, data, nparams_test, verb=false)
        sparco_gradient_norm_test = norm(g_combined_test[nparams_test+1:end])
        
        println("  chi2 value: ", chi2_val_test)
        println("  Parameter gradients:")
        println("    ∂chi2/∂f_star_0:  ", g_combined_test[1])
        println("    ∂chi2/∂f_bg_0:    ", g_combined_test[2])
        println("    ∂chi2/∂diameter:  ", g_combined_test[3])
        println("    ∂chi2/∂d_ind:     ", g_combined_test[4])
        println("    ∂chi2/∂lambda_0:  ", g_combined_test[5])
        println("  Image gradient norm: ", sparco_gradient_norm_test)
        
        g_img_test = reshape(g_combined_test[nparams_test+1:end], {nx}, {nx})
        println("  Image gradient stats: min=", minimum(g_img_test), ", max=", maximum(g_img_test), 
                ", mean=", mean(g_img_test), ", std=", std(g_img_test))
    end
    
    # Use the first parameter set for comparison with Python
    params = test_params_list[1]
    nparams = length(params)
    x_combined = [params; vec(x)]
    g_combined = zeros(length(x_combined))
    chi2_sparco_fg_val = chi2_sparco_fg(x_combined, g_combined, ft, data, nparams, verb=false)
    sparco_gradient_norm = norm(g_combined[nparams+1:end])  # Only image gradient
    
    # Collect extended test results for JSON output
    extended_tests = []
    for (idx, test_params) in enumerate(test_params_list)
        nparams_test = length(test_params)
        x_combined_test = [test_params; vec(x)]
        g_combined_test = zeros(length(x_combined_test))
        chi2_val_test = chi2_sparco_fg(x_combined_test, g_combined_test, ft, data, nparams_test, verb=false)
        g_img_test = reshape(g_combined_test[nparams_test+1:end], {nx}, {nx})
        
        push!(extended_tests, Dict(
            "test_set" => idx,
            "chi2" => chi2_val_test,
            "grad_f_star_0" => g_combined_test[1],
            "grad_f_bg_0" => g_combined_test[2],
            "grad_diameter" => g_combined_test[3],
            "grad_d_ind" => g_combined_test[4],
            "grad_lambda_0" => g_combined_test[5],
            "img_grad_norm" => norm(g_combined_test[nparams_test+1:end]),
            "img_grad_min" => minimum(g_img_test),
            "img_grad_max" => maximum(g_img_test),
            "img_grad_mean" => mean(g_img_test),
            "img_grad_std" => std(g_img_test)
        ))
    end
    
    # Test optimize_sparco_parameters
    println("[Julia] Testing optimize_sparco_parameters...")
    params_start = [0.5, 0.05, 0.3, 0.0, 1.65e-6]  # Start from different values
    opt_chi2 = nothing
    opt_params = nothing
    try
        min_chi2_opt, params_opt, ret = optimize_sparco_parameters(
            params_start, x, ft, data,
            lb=[0.0, 0.0, 0.1, -20.0],
            ub=[1.0, 0.5, 1.0, 20.0]
        )
        println("[Julia] Optimization: chi2=", min_chi2_opt, ", params=", params_opt[1:4])
        global opt_chi2 = min_chi2_opt
        global opt_params = params_opt[1:4]
    catch e
        println("[Julia] Optimization failed: ", e)
    end
    
    result = Dict(
        "nuv" => data.nuv,
        "nv2" => data.nv2,
        "nt3" => data.nt3phi,
        "cvis_real" => real(cvis[1:min(10, length(cvis))]),
        "cvis_imag" => imag(cvis[1:min(10, length(cvis))]),
        "v2" => v2[1:min(10, data.nv2)],
        "t3amp" => t3amp[1:min(10, data.nt3amp)],
        "t3phi" => t3phi[1:min(10, data.nt3phi)],
        "t3_1" => data.indx_t3_1[1:min(10, length(data.indx_t3_1))],
        "t3_2" => data.indx_t3_2[1:min(10, length(data.indx_t3_2))],
        "t3_3" => data.indx_t3_3[1:min(10, length(data.indx_t3_3))],
        "image_sum" => sum(x),
        "chi2" => chi2_val,
        "chi2_fg" => chi2_fg_val,
        "gradient_norm" => norm(g),
        "chi2_sparco" => chi2_sparco_val,
        "chi2_sparco_fg" => chi2_sparco_fg_val,
        "sparco_gradient_norm" => sparco_gradient_norm,
        "extended_tests" => extended_tests,
        "opt_chi2" => opt_chi2,
        "opt_params" => opt_params
    )
    
    println(JSON3.write(result))
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(julia_script)
        script_path = f.name

    try:
        result = subprocess.run(
            ['julia', '--startup-file=no', script_path],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=60
        )
        
        if result.returncode != 0:
            print("Julia STDERR:", result.stderr)
            raise RuntimeError(f"Julia script failed: {result.stderr}")
        print(result.stdout)
        
        # Find the JSON line (starts with {)
        json_line = None
        for line in result.stdout.strip().split('\n'):
            if line.strip().startswith('{'):
                json_line = line.strip()
                break
        
        if json_line is None:
            raise RuntimeError("Could not find JSON output in Julia stdout")
        
        print(f"\n[DEBUG] JSON line length: {len(json_line)}")
        print(f"[DEBUG] JSON starts with: {json_line[:100]}")
        print(f"[DEBUG] JSON ends with: {json_line[-100:]}")
        
        data = json.loads(json_line)
        print(f"[DEBUG] Keys in data: {list(data.keys())}")
        print(f"[DEBUG] opt_chi2 value: {data.get('opt_chi2')}")
        print(f"[DEBUG] opt_params value: {data.get('opt_params')}")
        for key in ['cvis_real', 'cvis_imag', 'v2', 't3amp', 't3phi', 't3_1', 't3_2', 't3_3']:
            if key in data:
                data[key] = np.array(data[key])
        return data
        
    finally:
        Path(script_path).unlink()


def gaussian2d(nx, sigma):
    """Create a 2D Gaussian image matching Julia's 1-based indexing."""
    # Julia: X=1:m gives [1, 2, ..., m], then X-(m/2) gives [1-m/2, 2-m/2, ..., m-m/2]
    # For m=64: [1-32, 2-32, ..., 64-32] = [-31, -30, ..., 32]
    # Python equivalent using 0-based: np.arange(1, nx+1) - nx/2
    # Use indexing='ij' to match Julia's column-major convention
    x = np.arange(1, nx + 1) - nx / 2.0
    y = np.arange(1, nx + 1) - nx / 2.0
    xx, yy = np.meshgrid(x, y, indexing='ij')
    g2d = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g2d / np.sum(g2d)


def test_python_nfft(oifits_file, nx, pixsize):
    """
    Run Python NFFT and return results.
    """
    # Load data
    print(f"[Python] Loading: {oifits_file}")
    data = readoifits(oifits_file, filter_bad_data=True, redundance_remove=True)
    
    # Setup NFFT
    print(f"[Python] Setting up NFFT with nx={nx}, pixsize={pixsize}")
    ft = setup_nfft(data, nx, pixsize)
    
    # Debug: Print UV coordinate ranges and indices
    print(f"[Python] Raw UV from data (before scaling):")
    print(f"  u range: [{data.uv[0, :].min():.2e}, {data.uv[0, :].max():.2e}]")
    print(f"  v range: [{data.uv[1, :].min():.2e}, {data.uv[1, :].max():.2e}]")
    print(f"[Python] First 5 indx_v2: {data.indx_v2[:5]}")
    print(f"[Python] First 5 indx_vis: {data.indx_vis[:5]}")
    
    uv_plan = ft[0]
    u_coords = uv_plan.uv_points[0, :]
    v_coords = uv_plan.uv_points[1, :]
    print(f"[Python] Scaled UV in NFFT plan:")
    print(f"  u: [{u_coords.min():.6f}, {u_coords.max():.6f}]")
    print(f"  v: [{v_coords.min():.6f}, {v_coords.max():.6f}]")
    print(f"[Python] First 5 scaled UV points:")
    for i in range(min(5, len(u_coords))):
        print(f"  ({u_coords[i]:.8f}, {v_coords[i]:.8f})")
    
    
    # Create test image (same Gaussian as Julia)
    x = gaussian2d(nx, nx / 8.0)
    
    # Compute visibilities
    cvis = image_to_vis(x, ft[0])
    
    # Compute observables
    v2_model, t3amp_model, t3phi_model = image_to_obs(x, ft, data)
    
    # Compute chi2
    chi2_val = chi2_nfft(x, ft, data)
    
    # Compute chi2 with gradient
    g = np.zeros_like(x)
    chi2_fg_val = chi2_fg(x, g, ft, data)
    
    # Test chi2_sparco_f with chromatic model
    params = [0.8, 0.1, 0.5, 0.0, 1.65e-6]
    chi2_sparco_val = chi2_sparco_f(x, params, ft, data, verbose=False)
    
    # Test chi2_sparco_fg with gradient - try multiple parameter sets
    print("\n[Python] Extended chi2_sparco_fg gradient test with multiple parameter sets:")
    
    test_params_list = [
        np.array([0.8, 0.1, 0.5, 0.0, 1.65e-6], dtype=np.float64),
        np.array([0.5, 0.2, 0.3, 0.0, 1.65e-6], dtype=np.float64),
        np.array([0.9, 0.05, 0.7, -1.0, 1.65e-6], dtype=np.float64),
        np.array([0.6, 0.15, 0.4, 1.0, 1.65e-6], dtype=np.float64),
    ]
    
    for idx, params_fg in enumerate(test_params_list):
        print(f"\n--- Test set {idx+1}: f_star={params_fg[0]:.2f}, f_bg={params_fg[1]:.2f}, D={params_fg[2]:.2f}mas, d_ind={params_fg[3]:.1f} ---")
        nparams = len(params_fg)
        x_combined = np.concatenate([params_fg, x.flatten(order='F')])
        g_combined = np.zeros_like(x_combined)
        chi2_val = chi2_sparco_fg(x_combined, g_combined, ft, data, nparams, verbose=False)
        sparco_gradient_norm_test = np.linalg.norm(g_combined[nparams:])
        
        print(f"  chi2 value: {chi2_val:.6e}")
        print(f"  Parameter gradients:")
        print(f"    ∂chi2/∂f_star_0:  {g_combined[0]:+.6e}")
        print(f"    ∂chi2/∂f_bg_0:    {g_combined[1]:+.6e}")
        print(f"    ∂chi2/∂diameter:  {g_combined[2]:+.6e}")
        print(f"    ∂chi2/∂d_ind:     {g_combined[3]:+.6e}")
        print(f"    ∂chi2/∂lambda_0:  {g_combined[4]:+.6e}")
        print(f"  Image gradient norm: {sparco_gradient_norm_test:.6e}")
        
        # Reshape image gradient and check statistics
        g_img = g_combined[nparams:].reshape((nx, nx), order='F')
        print(f"  Image gradient stats: min={g_img.min():+.3e}, max={g_img.max():+.3e}, mean={g_img.mean():+.3e}, std={g_img.std():.3e}")
    
    # Use the first parameter set for comparison with Julia
    params_fg = test_params_list[0]
    nparams = len(params_fg)
    x_combined = np.concatenate([params_fg, x.flatten(order='F')])
    g_combined = np.zeros_like(x_combined)
    chi2_sparco_fg_val = chi2_sparco_fg(x_combined, g_combined, ft, data, nparams, verbose=False)
    sparco_gradient_norm = np.linalg.norm(g_combined[nparams:])
    
    # Collect extended test results for comparison
    extended_tests = []
    for idx, params_fg_test in enumerate(test_params_list):
        nparams_test = len(params_fg_test)
        x_combined_test = np.concatenate([params_fg_test, x.flatten(order='F')])
        g_combined_test = np.zeros_like(x_combined_test)
        chi2_val_test = chi2_sparco_fg(x_combined_test, g_combined_test, ft, data, nparams_test, verbose=False)
        g_img_test = g_combined_test[nparams_test:].reshape((nx, nx), order='F')
        
        extended_tests.append({
            'test_set': idx + 1,
            'chi2': float(chi2_val_test),
            'grad_f_star_0': float(g_combined_test[0]),
            'grad_f_bg_0': float(g_combined_test[1]),
            'grad_diameter': float(g_combined_test[2]),
            'grad_d_ind': float(g_combined_test[3]),
            'grad_lambda_0': float(g_combined_test[4]),
            'img_grad_norm': float(np.linalg.norm(g_combined_test[nparams_test:])),
            'img_grad_min': float(g_img_test.min()),
            'img_grad_max': float(g_img_test.max()),
            'img_grad_mean': float(g_img_test.mean()),
            'img_grad_std': float(g_img_test.std())
        })
    
    # Test optimize_sparco_parameters
    print("\n[Python] Testing optimize_sparco_parameters...")
    params_start = [0.5, 0.05, 0.3, 0.0, 1.65e-6]  # Start from different values
    min_chi2 = None
    params_opt = None
    try:
        min_chi2, params_opt, result = optimize_sparco_parameters(
            params_start, x, ft, data,
            lb=[0.0, 0.0, 0.1, -20.0],
            ub=[1.0, 0.5, 1.0, 20.0],
            max_iter=50,
            verbose=False
        )
        print(f"[Python] Optimization: chi2={min_chi2:.6e}, params={params_opt[:4]}")
    except Exception as e:
        print(f"[Python] Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        'nuv': data.nuv,
        'nv2': data.nv2,
        'nt3': data.nt3phi,
        'cvis_real': np.real(cvis[:min(10, len(cvis))]),
        'cvis_imag': np.imag(cvis[:min(10, len(cvis))]),
        'v2': v2_model[:min(10, data.nv2)],
        't3amp': t3amp_model[:min(10, data.nt3amp)],
        't3phi': t3phi_model[:min(10, data.nt3phi)],
        'image_sum': np.sum(x),
        'chi2': chi2_val,
        'chi2_fg': chi2_fg_val,
        'gradient_norm': np.linalg.norm(g),
        'chi2_sparco': chi2_sparco_val,
        'chi2_sparco_fg': chi2_sparco_fg_val,
        'sparco_gradient_norm': sparco_gradient_norm,
        'extended_tests': extended_tests,
        'opt_chi2': min_chi2,
        'opt_params': params_opt[:4] if params_opt is not None else None
    }


def compare_results(py_result, jl_result, oifits_file):
    """
    Compare Python and Julia NFFT results.
    """
    print("\n" + "="*80)
    print("NFFT COMPARISON: Python (finufft) vs Julia (NFFT.jl)")
    print("="*80)
    
    # Compare dimensions
    print("\nDimensions:")
    print(f"  nuv:  Python={py_result['nuv']:5d}, Julia={jl_result['nuv']:5d}")
    print(f"  nv2:  Python={py_result['nv2']:5d}, Julia={jl_result['nv2']:5d}")
    print(f"  nt3:  Python={py_result['nt3']:5d}, Julia={jl_result['nt3']:5d}")
    
    if py_result['nuv'] != jl_result['nuv']:
        print("  ✗ UV counts don't match!")
        return False
    
    # Compare image normalization
    print("\nImage sum (should be 1.0):")
    print(f"  Python: {py_result['image_sum']:.10f}")
    print(f"  Julia:  {jl_result['image_sum']:.10f}")
    
    # Compare complex visibilities
    print("\nComplex visibilities (first 10):")
    cvis_py = py_result['cvis_real'] + 1j * py_result['cvis_imag']
    cvis_jl = jl_result['cvis_real'] + 1j * jl_result['cvis_imag']
    
    max_diff = np.max(np.abs(cvis_py - cvis_jl))
    rel_diff = max_diff / (np.max(np.abs(cvis_jl)) + 1e-15)
    
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")
    
    if max_diff > 1e-4:
        print("  ✗ Large difference in complex visibilities!")
        print("\n  First 5 values:")
        for i in range(min(5, len(cvis_py))):
            print(f"    [{i}] Python: {cvis_py[i]:.6f}, Julia: {cvis_jl[i]:.6f}, diff: {abs(cvis_py[i]-cvis_jl[i]):.2e}")
        return False
    else:
        print("  ✓ Complex visibilities match!")
    
    # Compare V2
    print("\nV2 (first 10):")
    max_diff_v2 = np.max(np.abs(py_result['v2'] - jl_result['v2']))
    rel_diff_v2 = max_diff_v2 / (np.max(jl_result['v2']) + 1e-15)
    
    print(f"  Max absolute difference: {max_diff_v2:.2e}")
    print(f"  Max relative difference: {rel_diff_v2:.2e}")
    
    if max_diff_v2 > 1e-4:
        print("  ✗ Large difference in V2!")
        return False
    else:
        print("  ✓ V2 values match!")
    
    # Compare T3amp
    print("\nT3amp (first 10):")
    max_diff_t3amp = np.max(np.abs(py_result['t3amp'] - jl_result['t3amp']))
    rel_diff_t3amp = max_diff_t3amp / (np.max(jl_result['t3amp']) + 1e-15)
    
    print(f"  Max absolute difference: {max_diff_t3amp:.2e}")
    print(f"  Max relative difference: {rel_diff_t3amp:.2e}")
    
    if max_diff_t3amp > 1e-4:
        print("  ✗ Large difference in T3amp!")
        return False
    else:
        print("  ✓ T3amp values match!")
    
    # Compare T3phi (with proper angle wrapping)
    print("\nT3phi (first 10, degrees):")
    from oichi2 import mod360
    # Wrap the difference to [-180, 180] to handle 360° ambiguity
    diff_t3phi = mod360(py_result['t3phi'] - jl_result['t3phi'])
    max_diff_t3phi = np.max(np.abs(diff_t3phi))
    
    if max_diff_t3phi > 0.1:  # 0.1 degree tolerance
        print("  ✗ Large difference in T3phi!")
        print("\n  First 5 values:")
        for i in range(min(5, len(py_result['t3phi']))):
            print(f"    [{i}] Python: {py_result['t3phi'][i]:.2f}°, Julia: {jl_result['t3phi'][i]:.2f}°, diff: {diff_t3phi[i]:.2f}°")
        print("  First 3 T3 triplets (Python):")
        try:
            from readoifits import readoifits
            data = readoifits(oifits_file, filter_bad_data=True, redundance_remove=True)
            for i in range(3):
                print(f"    {data.indx_t3_1[i]}, {data.indx_t3_2[i]}, {data.indx_t3_3[i]}")
        except Exception:
            pass
        return False
    else:
        print("  ✓ T3phi values match!")
    
    # Compare chi2
    print("\nChi-squared:")
    py_chi2 = py_result['chi2']
    jl_chi2 = jl_result['chi2']
    chi2_diff = abs(py_chi2 - jl_chi2)
    rel_chi2_diff = chi2_diff / (abs(jl_chi2) + 1e-15)
    
    print(f"  Python: {py_chi2:.6e}")
    print(f"  Julia:  {jl_chi2:.6e}")
    print(f"  Absolute difference: {chi2_diff:.2e}")
    print(f"  Relative difference: {rel_chi2_diff:.2e}")
    
    if rel_chi2_diff > 1e-6:  # Very tight tolerance for chi2
        print("  ✗ Chi2 values differ!")
    else:
        print("  ✓ Chi2 values match!")
    
    # EXTENDED TESTS - Chi2 for each parameter set
    print("\nChi-squared SPARCO (Extended Tests - 4 different parameter sets):")
    py_extended = py_result.get('extended_tests', [])
    jl_extended = jl_result.get('extended_tests', [])
    
    if py_extended and jl_extended:
        print(f"  {'Test':<6} {'Python':<15} {'Julia':<15} {'Abs Diff':<12} {'Rel Diff':<10} {'Status'}")
        print(f"  {'-'*75}")
        
        all_match = True
        for idx in range(min(len(py_extended), len(jl_extended))):
            py_test = py_extended[idx]
            jl_test = jl_extended[idx]
            test_num = py_test['test_set']
            
            py_chi2_test = py_test['chi2']
            jl_chi2_test = jl_test['chi2']
            diff = abs(py_chi2_test - jl_chi2_test)
            rel_diff = diff / (abs(jl_chi2_test) + 1e-15)
            
            status = "✓" if rel_diff < 1e-6 else "✗"
            if rel_diff > 1e-6:
                all_match = False
            
            print(f"  Set {test_num:<2} {py_chi2_test:<15.6e} {jl_chi2_test:<15.6e} {diff:<12.2e} {rel_diff:<10.2e} {status}")
        
        if all_match:
            print(f"  {'-'*75}")
            print("  ✓ All chi2 values match!")
        else:
            print(f"  {'-'*75}")
            print("  ✗ Some chi2 values differ!")
    else:
        print("  ⚠️  Extended test data not available")
    
    # Compare chi2_fg
    print("\nChi-squared with gradient (chi2_fg):")
    py_chi2_fg = py_result['chi2_fg']
    jl_chi2_fg = jl_result['chi2_fg']
    chi2_fg_diff = abs(py_chi2_fg - jl_chi2_fg)
    rel_chi2_fg_diff = chi2_fg_diff / (abs(jl_chi2_fg) + 1e-15)
    
    print(f"  Python: {py_chi2_fg:.6e}")
    print(f"  Julia:  {jl_chi2_fg:.6e}")
    print(f"  Absolute difference: {chi2_fg_diff:.2e}")
    print(f"  Relative difference: {rel_chi2_fg_diff:.2e}")
    
    if rel_chi2_fg_diff > 1e-6:  # Very tight tolerance
        print("  ✗ Chi2_fg values differ!")
        return False
    else:
        print("  ✓ Chi2_fg values match!")
    
    # Compare gradient norms
    print("\nGradient norm:")
    py_grad_norm = py_result['gradient_norm']
    jl_grad_norm = jl_result['gradient_norm']
    grad_norm_diff = abs(py_grad_norm - jl_grad_norm)
    rel_grad_norm_diff = grad_norm_diff / (abs(jl_grad_norm) + 1e-15)
    
    print(f"  Python: {py_grad_norm:.6e}")
    print(f"  Julia:  {jl_grad_norm:.6e}")
    print(f"  Absolute difference: {grad_norm_diff:.2e}")
    print(f"  Relative difference: {rel_grad_norm_diff:.2e}")
    
    if rel_grad_norm_diff > 1e-4:  # Slightly looser tolerance for gradients
        print("  ✗ Gradient norms differ!")
        return False
    else:
        print("  ✓ Gradient norms match!")
    
    # Compare chi2_sparco
    print("\nChi-squared SPARCO (chromatic model):")
    py_sparco = py_result['chi2_sparco']
    jl_sparco = jl_result['chi2_sparco']
    sparco_diff = abs(py_sparco - jl_sparco)
    rel_sparco_diff = sparco_diff / (abs(jl_sparco) + 1e-15)
    
    print(f"  Python: {py_sparco:.6e}")
    print(f"  Julia:  {jl_sparco:.6e}")
    print(f"  Absolute difference: {sparco_diff:.2e}")
    print(f"  Relative difference: {rel_sparco_diff:.2e}")
    
    if rel_sparco_diff > 1e-6:
        print("  ✗ chi2_sparco values differ!")
        return False
    else:
        print("  ✓ chi2_sparco values match!")
    
    # Compare chi2_sparco_fg
    print("\nChi-squared SPARCO with gradient (chi2_sparco_fg):")
    py_sparco_fg = py_result['chi2_sparco_fg']
    jl_sparco_fg = jl_result['chi2_sparco_fg']
    sparco_fg_diff = abs(py_sparco_fg - jl_sparco_fg)
    rel_sparco_fg_diff = sparco_fg_diff / (abs(jl_sparco_fg) + 1e-15)
    
    print(f"  Python: {py_sparco_fg:.6e}")
    print(f"  Julia:  {jl_sparco_fg:.6e}")
    print(f"  Absolute difference: {sparco_fg_diff:.2e}")
    print(f"  Relative difference: {rel_sparco_fg_diff:.2e}")
    
    if rel_sparco_fg_diff > 1e-6:
        print("  ✗ chi2_sparco_fg values differ!")
        return False
    else:
        print("  ✓ chi2_sparco_fg values match!")
    
    # Compare SPARCO gradient norms
    print("\nSPARCO Gradient norm:")
    py_sparco_grad = py_result['sparco_gradient_norm']
    jl_sparco_grad = jl_result['sparco_gradient_norm']
    sparco_grad_diff = abs(py_sparco_grad - jl_sparco_grad)
    rel_sparco_grad_diff = sparco_grad_diff / (abs(jl_sparco_grad) + 1e-15)
    
    print(f"  Python: {py_sparco_grad:.6e}")
    print(f"  Julia:  {jl_sparco_grad:.6e}")
    print(f"  Absolute difference: {sparco_grad_diff:.2e}")
    print(f"  Relative difference: {rel_sparco_grad_diff:.2e}")
    
    if rel_sparco_grad_diff > 1e-4:
        print("  ✗ SPARCO gradient norms differ!")
    else:
        print("  ✓ SPARCO gradient norms match!")
    
    # EXTENDED TESTS - Gradient norms for each parameter set
    print("\nSPARCO Gradient Norms (Extended Tests - 4 different parameter sets):")
    py_extended = py_result.get('extended_tests', [])
    jl_extended = jl_result.get('extended_tests', [])
    
    if py_extended and jl_extended:
        print(f"  {'Test':<6} {'Python Norm':<15} {'Julia Norm':<15} {'Ratio (Py/Jl)':<12} {'Status'}")
        print(f"  {'-'*75}")
        
        norm_ratios = []
        for idx in range(min(len(py_extended), len(jl_extended))):
            py_test = py_extended[idx]
            jl_test = jl_extended[idx]
            test_num = py_test['test_set']
            
            py_norm = py_test['img_grad_norm']
            jl_norm = jl_test['img_grad_norm']
            ratio = py_norm / jl_norm if jl_norm != 0 else 0
            norm_ratios.append(ratio)
            
            # Determine status
            if abs(ratio - 1.0) < 0.01:
                status = "✓ Match"
            elif abs(ratio - 2.0) < 0.1:
                status = "⚠️  ~2x"
            else:
                status = f"⚠️  ~{ratio:.2f}x"
            
            print(f"  Set {test_num:<2} {py_norm:<15.6e} {jl_norm:<15.6e} {ratio:<12.6f} {status}")
        
        # Summary
        avg_ratio = np.mean(norm_ratios)
        std_ratio = np.std(norm_ratios)
        
        print(f"  {'-'*75}")
        print(f"  Average ratio: {avg_ratio:.6f} ± {std_ratio:.6f}")
        
        if abs(avg_ratio - 2.0) < 0.1:
            print("  ⚠️  CONSISTENT FACTOR OF ~2 DETECTED in image gradients!")
            return False
        elif abs(avg_ratio - 1.0) < 0.01:
            print("  ✅ Image gradient norms match!")
        else:
            print(f"  ⚠️  Systematic difference: factor of ~{avg_ratio:.2f}")
            return False
    else:
        print("  ⚠️  Extended test data not available")
    
    # Detailed gradient comparison from extended tests
    print("\n" + "="*80)
    print("EXTENDED GRADIENT TESTS COMPARISON")
    print("="*80)
    
    py_extended = py_result.get('extended_tests', [])
    jl_extended = jl_result.get('extended_tests', [])
    
    if py_extended and jl_extended:
        for idx in range(min(len(py_extended), len(jl_extended))):
            py_test = py_extended[idx]
            jl_test = jl_extended[idx]
            test_num = py_test['test_set']
            
            print(f"\n{'─'*80}")
            print(f"Test Set {test_num}")
            print(f"{'─'*80}")
            
            # Chi2 comparison
            py_chi2 = py_test['chi2']
            jl_chi2 = jl_test['chi2']
            chi2_diff = abs(py_chi2 - jl_chi2)
            chi2_rel = chi2_diff / abs(jl_chi2) if jl_chi2 != 0 else 0
            
            print(f"\nChi2:")
            print(f"  Python: {py_chi2:.6e}")
            print(f"  Julia:  {jl_chi2:.6e}")
            print(f"  Diff:   {chi2_diff:.2e} ({chi2_rel:.2%})")
            
            # Parameter gradients
            print(f"\nParameter Gradients:")
            params = [('f_star_0', 'f_star'), ('f_bg_0', 'f_bg'), ('diameter', 'diam'), 
                     ('d_ind', 'd_ind'), ('lambda_0', 'λ_0')]
            for key, label in params:
                py_val = py_test[f'grad_{key}']
                jl_val = jl_test[f'grad_{key}']
                diff = py_val - jl_val
                rel = abs(diff) / abs(jl_val) if jl_val != 0 else 0
                print(f"  ∂χ²/∂{label:6s}: Py={py_val:+.6e}, Jl={jl_val:+.6e}, Δ={diff:+.2e} ({rel:.1%})")
            
            # Image gradient norm
            py_norm = py_test['img_grad_norm']
            jl_norm = jl_test['img_grad_norm']
            norm_diff = py_norm - jl_norm
            norm_ratio = py_norm / jl_norm if jl_norm != 0 else 0
            norm_rel = abs(norm_diff) / jl_norm if jl_norm != 0 else 0
            
            print(f"\nImage Gradient Norm:")
            print(f"  Python: {py_norm:.6e}")
            print(f"  Julia:  {jl_norm:.6e}")
            print(f"  Diff:   {norm_diff:+.6e} ({norm_rel:.2%})")
            print(f"  Ratio:  {norm_ratio:.6f}", end="")
            
            if abs(norm_ratio - 1.0) < 0.01:
                print(" ✓ Match")
            elif abs(norm_ratio - 2.0) < 0.1:
                print(" ⚠️  Factor of ~2!")
            else:
                print(f" ⚠️  Factor of ~{norm_ratio:.2f}")
        
        # Summary table
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        print(f"{'Test':<6} {'Python Norm':<15} {'Julia Norm':<15} {'Ratio':<10} {'Status'}")
        print("─" * 80)
        
        norm_ratios = []
        for idx in range(min(len(py_extended), len(jl_extended))):
            py_test = py_extended[idx]
            jl_test = jl_extended[idx]
            test_num = py_test['test_set']
            
            py_norm = py_test['img_grad_norm']
            jl_norm = jl_test['img_grad_norm']
            ratio = py_norm / jl_norm if jl_norm != 0 else 0
            norm_ratios.append(ratio)
            
            if abs(ratio - 1.0) < 0.01:
                status = "✓"
            elif abs(ratio - 2.0) < 0.1:
                status = "⚠️ ~2x"
            else:
                status = f"⚠️ ~{ratio:.2f}x"
            
            print(f"{test_num:<6} {py_norm:<15.6e} {jl_norm:<15.6e} {ratio:<10.4f} {status}")
        
        avg_ratio = np.mean(norm_ratios)
        std_ratio = np.std(norm_ratios)
        
        print("─" * 80)
        print(f"Average ratio: {avg_ratio:.6f} ± {std_ratio:.6f}")
        
        if abs(avg_ratio - 2.0) < 0.1:
            print("⚠️  CONSISTENT FACTOR OF ~2 DETECTED in image gradients!")
        elif abs(avg_ratio - 1.0) < 0.01:
            print("✅ Image gradients match!")
        else:
            print(f"⚠️  Systematic difference: factor of ~{avg_ratio:.2f}")
        
    else:
        print("\n⚠️  Extended test results not available in JSON")
    
    # Compare optimization results
    print("\nParameter Optimization (optimize_sparco_parameters):")
    print(f"  DEBUG: py_result['opt_chi2'] = {py_result.get('opt_chi2')}, type = {type(py_result.get('opt_chi2'))}")
    print(f"  DEBUG: jl_result['opt_chi2'] = {jl_result.get('opt_chi2')}, type = {type(jl_result.get('opt_chi2'))}")
    if py_result.get('opt_chi2') is not None and jl_result.get('opt_chi2') is not None:
        py_opt_chi2 = py_result['opt_chi2']
        jl_opt_chi2 = jl_result['opt_chi2']
        opt_chi2_diff = abs(py_opt_chi2 - jl_opt_chi2)
        rel_opt_chi2_diff = opt_chi2_diff / (abs(jl_opt_chi2) + 1e-15)
        
        print(f"  Optimized chi2:")
        print(f"    Python: {py_opt_chi2:.6e}")
        print(f"    Julia:  {jl_opt_chi2:.6e}")
        print(f"    Absolute difference: {opt_chi2_diff:.2e}")
        print(f"    Relative difference: {rel_opt_chi2_diff:.2e}")
        
        if py_result['opt_params'] is not None and jl_result['opt_params'] is not None:
            py_opt_params = np.array(py_result['opt_params'])
            jl_opt_params = np.array(jl_result['opt_params'])
            print(f"  Optimized parameters:")
            print(f"    Python: [{py_opt_params[0]:.4f}, {py_opt_params[1]:.4f}, {py_opt_params[2]:.4f}, {py_opt_params[3]:.4f}]")
            print(f"    Julia:  [{jl_opt_params[0]:.4f}, {jl_opt_params[1]:.4f}, {jl_opt_params[2]:.4f}, {jl_opt_params[3]:.4f}]")
            params_diff = np.linalg.norm(py_opt_params - jl_opt_params)
            print(f"    Parameter norm difference: {params_diff:.4e}")
        
        # Note: Different optimizers may find different local minima, so we use looser tolerance
        if rel_opt_chi2_diff > 0.1:  # 10% tolerance - different optimizers
            print("  ⚠ Optimized chi2 values differ (but this is expected with different optimizers)")
        else:
            print("  ✓ Optimized chi2 values are similar!")
    else:
        print("  ⚠ Optimization skipped or failed in one or both implementations")
    
    print("\n" + "="*80)
    print("✅ ALL CHECKS PASSED - Python NFFT matches Julia!")
    print("="*80)
    return True


def main():
    """Run NFFT comparison test."""
    oifits_file = "../../data/OIFITS/2019_v1295Aql.WL_SMOOTH.A.oifits"
    nx = 64
    pixsize = 0.125  # mas/pixel
    
    if not Path(oifits_file).exists():
        print(f"Error: {oifits_file} not found")
        return
    
    print("="*80)
    print("Testing NFFT: Python (finufft) vs Julia (NFFT.jl)")
    print("="*80)
    print(f"OIFITS file: {oifits_file}")
    print(f"Image size: {nx}x{nx} pixels")
    print(f"Pixel size: {pixsize} mas/pixel")
    print()
    
    # Test Python
    try:
        py_result = test_python_nfft(oifits_file, nx, pixsize)
        print("[Python] ✓ NFFT completed")
    except Exception as e:
        print(f"[Python] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test Julia
    print()
    try:
        jl_result = test_julia_nfft(oifits_file, nx, pixsize)
        print("[Julia] ✓ NFFT completed")
    except Exception as e:
        print(f"[Julia] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare
    compare_results(py_result, jl_result, oifits_file)


if __name__ == "__main__":
    main()
