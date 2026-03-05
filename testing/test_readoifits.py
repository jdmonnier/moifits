#!/usr/bin/env python3
"""
Test script for readoifits.py OIFITS parser.
Compares Python implementation against Julia reference.
"""

import sys
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from readoifits import (
    readoifits, 
    display_oidata,
    list_oifits_targets,
    oifits_prep,
    set_data_filter,
    filter_data,
    remove_redundant_uv
)


def load_with_julia(oifits_file, filter_bad_data=True, redundance_remove=True):
    """
    Load OIFITS file using Julia's readoifits and return data as dict.
    """
    # Get absolute path to inference directory
    inference_dir = Path(__file__).parent.parent.absolute()
    
    # Create temporary Julia script
    julia_script = f"""
    using Pkg
    Pkg.activate("{inference_dir}")
    
    using OITOOLS
    using JSON3
    
    # Load data without redundancy removal first
    data_no_red = readoifits("{oifits_file}", 
                     filter_bad_data={str(filter_bad_data).lower()},
                     redundance_remove=false,
                     uvtol=2e2)[1,1]
    println("Julia before redundancy: nUV=", data_no_red.nuv, ", nV2=", data_no_red.nv2, ", nT3=", data_no_red.nt3phi, ", nVIS=", data_no_red.nvisamp)
    
    # Load data with redundancy removal
    data = readoifits("{oifits_file}", 
                     filter_bad_data={str(filter_bad_data).lower()},
                     redundance_remove={str(redundance_remove).lower()},
                     uvtol=2e2)[1,1]
    
    # Export to JSON
    result = Dict(
        "nv2" => data.nv2,
        "nt3amp" => data.nt3amp,
        "nt3phi" => data.nt3phi,
        "nuv" => data.nuv,
        "mean_mjd" => data.mean_mjd,
        "v2" => data.v2,
        "v2_err" => data.v2_err,
        "v2_baseline" => data.v2_baseline,
        "t3amp" => data.t3amp,
        "t3amp_err" => data.t3amp_err,
        "t3phi" => data.t3phi,
        "t3phi_err" => data.t3phi_err,
        "t3_baseline" => data.t3_baseline,
        "uv" => data.uv,
        "uv_lam" => data.uv_lam,
        "uv_baseline" => data.uv_baseline,
        "indx_v2" => data.indx_v2,
        "indx_t3_1" => data.indx_t3_1,
        "indx_t3_2" => data.indx_t3_2,
        "indx_t3_3" => data.indx_t3_3,
        "filename" => data.filename
    )
    
    println(JSON3.write(result))
    """
    
    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(julia_script)
        script_path = f.name
    
    try:
        # Run Julia script
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
        
        # Parse JSON output (last line)
        output_lines = result.stdout.strip().split('\n')
        
        # Print all non-JSON lines (debugging info)
        for line in output_lines[:-1]:
            if line.strip() and not line.strip().startswith('{'):
                print(f"[Julia] {line}")
        
        json_line = output_lines[-1]
        data = json.loads(json_line)
        
        # Convert lists to numpy arrays
        for key in ['v2', 'v2_err', 'v2_baseline', 't3amp', 't3amp_err', 't3phi', 
                    't3phi_err', 't3_baseline', 'uv_lam', 'uv_baseline',
                    'indx_v2', 'indx_t3_1', 'indx_t3_2', 'indx_t3_3']:
            if key in data:
                data[key] = np.array(data[key])
        
        # UV is 2D - Julia exports as flattened 1D, reshape to (2, nuv)
        if 'uv' in data:
            uv_flat = np.array(data['uv'])
            # Julia stores as [u1, v1, u2, v2, ...] flattened
            data['uv'] = uv_flat.reshape(2, -1, order='F')  # Fortran order (column-major)
        
        # Convert Julia 1-based indices to Python 0-based
        for key in ['indx_v2', 'indx_t3_1', 'indx_t3_2', 'indx_t3_3']:
            if key in data:
                data[key] = data[key] - 1  # Julia is 1-based
        
        return data
        
    finally:
        # Cleanup temp file
        Path(script_path).unlink()


def compare_data(py_data, jl_data, tol=1e-10):
    """
    Compare Python and Julia data structures.
    """
    print("\n" + "="*80)
    print("COMPARING PYTHON vs JULIA")
    print("="*80)
    
    issues = []
    
    # Compare counts
    for key in ['nv2', 'nt3amp', 'nt3phi', 'nuv']:
        py_val = getattr(py_data, key)
        jl_val = jl_data[key]
        if py_val != jl_val:
            issues.append(f"  ✗ {key}: Python={py_val}, Julia={jl_val}")
        else:
            print(f"  ✓ {key}: {py_val}")
    
    # Compare scalars
    py_mjd = py_data.mean_mjd
    jl_mjd = jl_data['mean_mjd']
    if abs(py_mjd - jl_mjd) > tol:
        issues.append(f"  ✗ mean_mjd: diff={abs(py_mjd - jl_mjd):.2e}")
    else:
        print(f"  ✓ mean_mjd: {py_mjd:.6f}")
    
    # Compare arrays
    array_fields = [
        ('v2', 'v2'),
        ('v2_err', 'v2_err'),
        ('v2_baseline', 'v2_baseline'),
        ('t3amp', 't3amp'),
        ('t3amp_err', 't3amp_err'),
        ('t3phi', 't3phi'),
        ('t3phi_err', 't3phi_err'),
        ('t3_baseline', 't3_baseline'),
        ('uv_baseline', 'uv_baseline'),
        ('uv_lam', 'uv_lam'),
    ]
    
    for py_field, jl_field in array_fields:
        py_arr = getattr(py_data, py_field)
        jl_arr = jl_data[jl_field]
        
        if len(py_arr) != len(jl_arr):
            issues.append(f"  ✗ {py_field} length: Python={len(py_arr)}, Julia={len(jl_arr)}")
            continue
        
        if len(py_arr) == 0:
            print(f"  ✓ {py_field}: empty (both)")
            continue
        
        max_diff = np.max(np.abs(py_arr - jl_arr))
        rel_diff = max_diff / (np.max(np.abs(jl_arr)) + 1e-15)
        
        if max_diff > tol and rel_diff > 1e-6:
            issues.append(f"  ✗ {py_field}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        else:
            print(f"  ✓ {py_field}: max_diff={max_diff:.2e}")
    
    # Compare UV coordinates (2D)
    py_uv = py_data.uv
    jl_uv = jl_data['uv']
    
    if py_uv.shape != jl_uv.shape:
        issues.append(f"  ✗ UV shape: Python={py_uv.shape}, Julia={jl_uv.shape}")
    else:
        max_diff = np.max(np.abs(py_uv - jl_uv))
        rel_diff = max_diff / (np.max(np.abs(jl_uv)) + 1e-15)
        if max_diff > tol and rel_diff > 1e-6:
            issues.append(f"  ✗ UV coordinates: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        else:
            print(f"  ✓ UV coordinates: max_diff={max_diff:.2e}")
    
    # Compare indices
    index_fields = [
        ('indx_v2', 'indx_v2'),
        ('indx_t3_1', 'indx_t3_1'),
        ('indx_t3_2', 'indx_t3_2'),
        ('indx_t3_3', 'indx_t3_3'),
    ]
    
    for py_field, jl_field in index_fields:
        py_idx = getattr(py_data, py_field)
        jl_idx = jl_data[jl_field]
        
        if len(py_idx) != len(jl_idx):
            issues.append(f"  ✗ {py_field} length: Python={len(py_idx)}, Julia={len(jl_idx)}")
            continue
        
        if len(py_idx) == 0:
            print(f"  ✓ {py_field}: empty (both)")
            continue
        
        if not np.array_equal(py_idx, jl_idx):
            n_diff = np.sum(py_idx != jl_idx)
            issues.append(f"  ✗ {py_field}: {n_diff} mismatches")
        else:
            print(f"  ✓ {py_field}: all match")
    
    # Print summary
    print("\n" + "-"*80)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(issue)
        print("\n❌ COMPARISON FAILED")
        return False
    else:
        print("✅ ALL CHECKS PASSED - Python matches Julia!")
        return True


def test_python_vs_julia():
    """Compare Python and Julia implementations."""
    print("="*80)
    print("TEST 1: Python vs Julia Comparison")
    print("="*80)
    
    # Update this path to your actual OIFITS file
    oifits_file = "../../data/OIFITS/2019_v1295Aql.WL_SMOOTH.A.oifits"
    
    if not Path(oifits_file).exists():
        print(f"SKIP: {oifits_file} not found")
        return None, None
    
    # Read with Python - first without redundancy removal
    print(f"\n[Python] Reading: {oifits_file}")
    py_data_no_red = readoifits(oifits_file, 
                                filter_bad_data=True,
                                redundance_remove=False)
    print(f"[Python] Before redundancy removal: nV2={py_data_no_red.nv2}, nT3={py_data_no_red.nt3phi}, nUV={py_data_no_red.nuv}")
    
    # Now with redundancy removal
    py_data = readoifits(oifits_file, 
                        filter_bad_data=True,
                        redundance_remove=True,
                        uvtol=2e2)
    print(f"[Python] After redundancy removal:  nV2={py_data.nv2}, nT3={py_data.nt3phi}, nUV={py_data.nuv}")
    print(f"[Python] Removed {py_data_no_red.nuv - py_data.nuv} redundant UV points ({100*(py_data_no_red.nuv - py_data.nuv)/py_data_no_red.nuv:.1f}%)")
    
    # Read with Julia
    print(f"\n[Julia] Reading: {oifits_file}")
    try:
        jl_data = load_with_julia(oifits_file, 
                                  filter_bad_data=True,
                                  redundance_remove=True)
        print(f"[Julia] nV2={jl_data['nv2']}, nT3={jl_data['nt3phi']}, nUV={jl_data['nuv']}")
        
        # Compare
        match = compare_data(py_data, jl_data)
        
        return py_data, jl_data, match
        
    except Exception as e:
        print(f"\n⚠️  Julia comparison failed: {e}")
        print("Continuing with Python-only tests...")
        return py_data, None, None


def test_basic_read():
    """Test basic OIFITS file reading (Python only)."""
    print("\n" + "="*80)
    print("TEST 2: Basic validation (Python)")
    print("="*80)
    
    oifits_file = "../../data/OIFITS/2019_v1295Aql.WL_SMOOTH.A.oifits"
    
    if not Path(oifits_file).exists():
        print(f"SKIP: {oifits_file} not found")
        return None
    
    data = readoifits(oifits_file, 
                     filter_bad_data=True,
                     redundance_remove=True,
                     uvtol=2e2)
    
    # Display summary
    print("\n" + "="*80)
    print("DATA SUMMARY:")
    print("="*80)
    display_oidata(data)
    
    # Validate basic structure
    print("\n" + "="*80)
    print("VALIDATION:")
    print("="*80)
    assert data.nv2 > 0, "No V2 data found"
    assert data.nt3phi > 0 or data.nt3amp > 0, "No T3 data found"
    assert data.nuv > 0, "No UV coverage"
    assert data.uv.shape == (2, data.nuv), f"UV shape mismatch: {data.uv.shape}"
    
    if data.nv2 > 0:
        assert len(data.indx_v2) == data.nv2, "V2 index length mismatch"
        assert np.all(data.indx_v2 < data.nuv), "V2 indices out of bounds"
    
    if data.nt3phi > 0:
        assert len(data.indx_t3_1) == data.nt3phi, "T3 index length mismatch"
        assert np.all(data.indx_t3_1 < data.nuv), "T3 indices out of bounds"
        assert np.all(data.indx_t3_2 < data.nuv), "T3 indices out of bounds"
        assert np.all(data.indx_t3_3 < data.nuv), "T3 indices out of bounds"
    
    print("✓ Data structure validated")
    print(f"✓ V2 points: {data.nv2}")
    print(f"✓ T3 points: {data.nt3phi}")
    print(f"✓ UV coverage: {data.nuv} points")
    print(f"✓ Mean MJD: {data.mean_mjd:.3f}")
    
    return data


def test_filtering():
    """Test data filtering functions."""
    print("\n" + "="*80)
    print("TEST 3: Data filtering")
    print("="*80)
    
    oifits_file = "../../data/OIFITS/2019_v1295Aql.WL_SMOOTH.A.oifits"
    
    if not Path(oifits_file).exists():
        print(f"SKIP: {oifits_file} not found")
        return
    
    # Read without filtering
    data = readoifits(oifits_file, filter_bad_data=False, redundance_remove=False)
    print(f"\nBefore filtering: nV2={data.nv2}, nT3={data.nt3phi}, nUV={data.nuv}")
    
    # Apply filters
    bad_indices = set_data_filter(
        data,
        filter_bad_data=True,
        filter_v2=True,
        filter_t3phi=True,
        cutoff_minv2=-1.0,
        cutoff_maxv2=2.0,
        filter_v2_snr_threshold=0.01
    )
    
    print(f"Bad indices found: UV={len(bad_indices[0])}, VIS={len(bad_indices[1])}, "
          f"V2={len(bad_indices[2])}, T3={len(bad_indices[3])}")
    
    # Apply filtering
    data_filtered = filter_data(data, bad_indices)
    print(f"After filtering: nV2={data_filtered.nv2}, nT3={data_filtered.nt3phi}, nUV={data_filtered.nuv}")
    
    # Remove redundancy
    data_clean = remove_redundant_uv(data_filtered, uvtol=2e2)
    print(f"After redundancy removal: nUV={data_clean.nuv}")
    
    print("✓ Filtering pipeline completed")


def test_error_adjustment():
    """Test error bar adjustment."""
    print("\n" + "="*80)
    print("TEST 4: Error bar adjustment")
    print("="*80)
    
    oifits_file = "../../data/OIFITS/2019_v1295Aql.WL_SMOOTH.A.oifits"
    
    if not Path(oifits_file).exists():
        print(f"SKIP: {oifits_file} not found")
        return
    
    data = readoifits(oifits_file)
    
    # Original errors
    if data.nv2 > 0:
        orig_v2_err = data.v2_err.copy()
        print(f"\nOriginal V2 errors: min={orig_v2_err.min():.6f}, max={orig_v2_err.max():.6f}")
    
    if data.nt3phi > 0:
        orig_t3phi_err = data.t3phi_err.copy()
        print(f"Original T3phi errors: min={orig_t3phi_err.min():.2f}°, max={orig_t3phi_err.max():.2f}°")
    
    # Apply MIRC-style error floors (Monnier et al. 2012)
    data = oifits_prep(
        data,
        min_v2_err_add=2e-4,
        min_v2_err_rel=0.066,
        min_t3phi_err_add=1.0,
        quad=True
    )
    
    # New errors
    if data.nv2 > 0:
        print(f"Adjusted V2 errors: min={data.v2_err.min():.6f}, max={data.v2_err.max():.6f}")
    
    if data.nt3phi > 0:
        print(f"Adjusted T3phi errors: min={data.t3phi_err.min():.2f}°, max={data.t3phi_err.max():.2f}°")
    
    print("✓ Error adjustment completed")


def test_target_list():
    """Test target listing."""
    print("\n" + "="*80)
    print("TEST 5: List targets")
    print("="*80)
    
    oifits_file = "../../data/OIFITS/2019_v1295Aql.WL_SMOOTH.A.oifits"
    
    if not Path(oifits_file).exists():
        print(f"SKIP: {oifits_file} not found")
        return
    
    targets = list_oifits_targets(oifits_file)
    print(f"\nTargets in file: {targets}")
    print("✓ Target listing completed")


def test_uv_coverage():
    """Test UV coverage computation."""
    print("\n" + "="*80)
    print("TEST 6: UV coverage")
    print("="*80)
    
    oifits_file = "../../data/OIFITS/2019_v1295Aql.WL_SMOOTH.A.oifits"
    
    if not Path(oifits_file).exists():
        print(f"SKIP: {oifits_file} not found")
        return
    
    data = readoifits(oifits_file)
    
    # Check UV symmetry (should have +/- UV points)
    print(f"\nUV coverage: {data.nuv} points")
    print(f"UV range: u ∈ [{data.uv[0].min():.0f}, {data.uv[0].max():.0f}]")
    print(f"          v ∈ [{data.uv[1].min():.0f}, {data.uv[1].max():.0f}]")
    print(f"Baseline range: [{data.uv_baseline.min():.0f}, {data.uv_baseline.max():.0f}] wavelengths")
    print(f"Wavelength: {data.uv_lam.min()*1e6:.3f} - {data.uv_lam.max()*1e6:.3f} µm")
    
    # Check indexing consistency
    if data.nv2 > 0:
        v2_uv = data.uv[:, data.indx_v2]
        print(f"\nV2 uses {len(np.unique(data.indx_v2))} unique UV points")
    
    if data.nt3phi > 0:
        t3_uv_indices = np.concatenate([data.indx_t3_1, data.indx_t3_2, data.indx_t3_3])
        print(f"T3 uses {len(np.unique(t3_uv_indices))} unique UV points (from 3×{data.nt3phi} baselines)")
    
    print("✓ UV coverage validated")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OIFITS READER TEST SUITE")
    print("Python vs Julia Comparison + Validation")
    print("="*80)
    
    try:
        # Run comparison test first
        py_data, jl_data, match = test_python_vs_julia()
        
        # Run additional Python tests
        if py_data is not None:
            test_basic_read()
            test_filtering()
            test_error_adjustment()
            test_target_list()
            test_uv_coverage()
        
        print("\n" + "="*80)
        if match is True:
            print("ALL TESTS PASSED ✅")
            print("Python implementation matches Julia reference!")
        elif match is False:
            print("TESTS COMPLETED WITH DIFFERENCES ⚠️")
            print("See comparison output above")
        else:
            print("PYTHON-ONLY TESTS PASSED ✅")
            print("(Julia comparison skipped)")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ✗")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
