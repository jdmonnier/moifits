import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import sys
import emcee
from pathlib import Path
import corner
import matplotlib.pyplot as plt

# Allow running this example directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moifits.readoifits import readoifits
from moifits.oichi2 import setup_nfft
from moifits.oioptimize import chi2_sparco_f
from moifits.fitting import (
    DISK_SPARCO_PARAM_NAMES,
    log_posterior_z,
    render_disk,
    unpack_params,
)

def run_mcmc_model(file_path=None, nx=64, pixsize=0.125, n_walkers=32, n_steps=10000):
    """
    Sets up data and FT plan, then runs MCMC to fit parametric disk+SPARCO model.
    Args:
        file_path: path to OIFITS file
        nx: image size (pixels)
        pixsize: pixel scale (mas/pixel)
        n_walkers: number of MCMC walkers
        n_steps: number of MCMC steps
    """

    data = readoifits(str(file_path), filter_bad_data=True, redundance_remove=True)
    ft = setup_nfft(data, nx, pixsize)
        
    weights = (1.0, 0.0, 1.0)  # relative weights for v2, t3amp, t3phi
        
    # Fixed SPARCO parameters only
    fixed_params = {
        'f_bg_0': 0.0,
        'diameter': 0.0,
        'd_ind': 4.0,
        'lambda_0': 1.6e-6,
    }
    print(f"Fixed params: {fixed_params}")
    
    # Initial free params: [inc, phi, radius, thickness, contrast, sigma, f_star_0]
    z0 = np.array([
        0.1,    # inc (rad)
        0.1,    # phi (rad)
        2.0,    # radius (mas) - start larger based on data
        1.0,    # thickness (mas) - start larger
        0.9,    # contrast - start high
        0.02,   # sigma (mas) - edge softness
        0.444,  # f_star_0
    ], dtype=np.float64)
    
    print(f"Initial free params: inc={z0[0]:.3f}, phi={z0[1]:.3f}, radius={z0[2]:.3f}, thickness={z0[3]:.3f}, contrast={z0[4]:.3f}, sigma={z0[5]:.3f}, f_star_0={z0[6]:.3f}")

    # Create grid for image rendering (in mas)
    x = np.linspace(-0.5 * nx * pixsize,
                    0.5 * nx * pixsize, nx)
    y = np.linspace(-0.5 * nx * pixsize,
                    0.5 * nx * pixsize, nx)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

    # Set up MCMC sampler (emcee-based)
    def log_prob_fn(z):
        """Log probability for emcee"""
        return log_posterior_z(z, x_grid, y_grid, ft, data, fixed_params,
                               weights=weights,
                               apply_cutoff=True,
                               normalize_flux=True)
    
    # Initialize walkers around z0 with small perturbations
    np.random.seed(42)
    noise_scale = 0.01
    p0 = z0 + noise_scale * np.random.randn(n_walkers, len(z0))
    # Create sampler and run
    sampler = emcee.EnsembleSampler(n_walkers, len(z0), log_prob_fn)
    print(f"Running {n_steps} steps with {n_walkers} walkers...")
    sampler.run_mcmc(p0, n_steps, progress=True)
    
    return sampler, data, ft, fixed_params

def plot_posterior(sampler, burn_in=100, thin=10, save_path=None):
    """
    Create corner plot of posterior samples.
    
    Args:
        sampler: emcee sampler output
        burn_in: number of initial steps to discard
        thin: thinning factor (keep every Nth sample)
        save_path: optional path to save figure
    
    Returns:
        fig: matplotlib figure
    """
    # Extract chain from emcee sampler
    # emcee returns (nwalkers, nsteps, ndim), need to transpose to (nsteps, nwalkers, ndim)
    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    
    # Discard burn-in and thin
    chain = chain[burn_in::thin, :, :]
    
    # Flatten walkers
    flat_samples = chain.reshape(-1, chain.shape[-1])
    
    # Parameter names (7 free parameters)
    labels = [
        r"$i$ [rad]",           # inclination
        r"$\phi$ [rad]",        # position angle
        r"$R$ [mas]",           # central radius
        r"$\Delta R$ [mas]",    # thickness
        r"$c$",                 # contrast
        r"$\sigma$ [mas]",      # edge softness
        r"$f_{\star,0}$",       # stellar flux fraction
    ]
    
    # Create corner plot
    fig = corner.corner(
        flat_samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={"fontsize": 10},
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved corner plot to {save_path}")
    
    return fig

def plot_chains(sampler, save_path=None):
    """
    Plot MCMC chains for all parameters (the "hairy caterpillar").
    
    Args:
        sampler: emcee sampler output
        save_path: optional path to save figure
    
    Returns:
        fig: matplotlib figure
    """
    # Get chain from emcee sampler
    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    
    # Parameter names
    labels = [
        r"$i$ [rad]",           # inclination
        r"$\phi$ [rad]",        # position angle
        r"$R$ [mas]",           # central radius
        r"$\Delta R$ [mas]",    # thickness
        r"$c$",                 # contrast
        r"$\sigma$ [mas]",      # edge softness
        r"$f_{\star,0}$",       # stellar flux fraction
    ]
    
    ndim = chain.shape[2]
    
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2*ndim), sharex=True)
    
    for i in range(ndim):
        ax = axes[i]
        # Plot all walkers
        ax.plot(chain[:, :, i], "k", alpha=0.3, linewidth=0.5)
        ax.set_ylabel(labels[i], fontsize=12)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel("Step Number", fontsize=12)
    fig.suptitle("MCMC Chains (Hairy Caterpillar)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved chain plot to {save_path}")
    
    return fig

def get_best_fit(sampler, burn_in=1000):
    """
    Get best-fit parameters from MCMC chain.
    
    Args:
        sampler: emcee sampler output
        burn_in: number of initial steps to discard
    
    Returns:
        best_params: parameter vector at maximum log probability
        median_params: median of posterior
        std_params: standard deviation of posterior
    """
    # Get chain and log probability from emcee
    chain = sampler.get_chain()  # (nsteps, nwalkers, ndim)
    log_prob = sampler.get_log_prob()  # (nsteps, nwalkers)
    
    # Discard burn-in
    chain = chain[burn_in:, :, :]
    log_prob = log_prob[burn_in:, :]
    
    # Find best fit
    flat_log_prob = log_prob.flatten()
    flat_chain = chain.reshape(-1, chain.shape[-1])
    best_idx = np.argmax(flat_log_prob)
    best_params = flat_chain[best_idx]
    
    # Get median and std
    median_params = np.median(flat_chain, axis=0)
    std_params = np.std(flat_chain, axis=0)
    
    return best_params, median_params, std_params

def plot_disk_images(sampler, fixed_params, nx=64, pixsize=0.125, burn_in=1000, save_path=None):
    """
    Plot best-fit, median, and 16-84% uncertainty images.
    
    Args:
        sampler: emcee sampler output
        fixed_params: dict with fixed SPARCO parameters
        nx: image size in pixels
        pixsize: pixel scale in mas/pixel
        burn_in: number of initial steps to discard
        save_path: optional path to save figure (without extension)
    """
    # Get parameters
    best_params, median_params, std_params = get_best_fit(sampler, burn_in=burn_in)
    
    # Get chain for percentiles
    chain = sampler.get_chain()
    chain = chain[burn_in:, :, :]
    flat_chain = chain.reshape(-1, chain.shape[-1])
    
    # Compute 16th and 84th percentile parameters
    p16_params = np.percentile(flat_chain, 16, axis=0)
    p84_params = np.percentile(flat_chain, 84, axis=0)
    
    # Create grid (in mas)
    x = np.linspace(-0.5 * nx * pixsize,
                    0.5 * nx * pixsize, nx)
    y = np.linspace(-0.5 * nx * pixsize,
                    0.5 * nx * pixsize, nx)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    
    # Render images
    def render_from_params(params):
        theta, _ = unpack_params(params, fixed_params)
        return render_disk(theta, x_grid, y_grid, apply_cutoff=True, normalize_flux=True)
    
    img_best = render_from_params(best_params)
    img_median = render_from_params(median_params)
    img_p16 = render_from_params(p16_params)
    img_p84 = render_from_params(p84_params)
    
    # Compute uncertainty image
    img_uncertainty = img_p84 - img_p16
    
    # Plot
    extent_mas = pixsize * nx / 2
    extent = [-extent_mas, extent_mas, -extent_mas, extent_mas]
    
    # import matplotlib
    # matplotlib.use('Agg')  # Use non-interactive backend
    # import matplotlib.pyplot as plt
    # plt.ioff()  # Turn off interactive mode
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Best fit
    im0 = axes[0, 0].imshow(img_best, extent=extent, origin='lower', cmap='viridis')
    axes[0, 0].set_title('Best Fit (Max Log Prob)')
    axes[0, 0].set_xlabel('RA offset (mas)')
    axes[0, 0].set_ylabel('Dec offset (mas)')
    plt.colorbar(im0, ax=axes[0, 0], label='Intensity')
    
    # Median
    im1 = axes[0, 1].imshow(img_median, extent=extent, origin='lower', cmap='viridis')
    axes[0, 1].set_title('Median')
    axes[0, 1].set_xlabel('RA offset (mas)')
    axes[0, 1].set_ylabel('Dec offset (mas)')
    plt.colorbar(im1, ax=axes[0, 1], label='Intensity')
    
    # 16th percentile
    im2 = axes[1, 0].imshow(img_p16, extent=extent, origin='lower', cmap='viridis')
    axes[1, 0].set_title('16th Percentile')
    axes[1, 0].set_xlabel('RA offset (mas)')
    axes[1, 0].set_ylabel('Dec offset (mas)')
    plt.colorbar(im2, ax=axes[1, 0], label='Intensity')
    
    # Uncertainty (84th - 16th)
    im3 = axes[1, 1].imshow(img_uncertainty, extent=extent, origin='lower', cmap='RdBu_r')
    axes[1, 1].set_title('Uncertainty (84th - 16th)')
    axes[1, 1].set_xlabel('RA offset (mas)')
    axes[1, 1].set_ylabel('Dec offset (mas)')
    plt.colorbar(im3, ax=axes[1, 1], label='Intensity Difference')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(f"{save_path}_images.png", dpi=150, bbox_inches='tight')
        print(f"Saved image comparison to {save_path}_images.png")
    plt.close(fig)  # Always close the figure to suppress display
    return fig, (img_best, img_median, img_p16, img_p84, img_uncertainty)

def main():
    """
    Main function to run MCMC fit and visualize results.
    """
    # MCMC configuration
    n_walkers = 16
    n_steps = 10000
    burn_in = 1000
    
    # Path to OIFITS file
    oifits_file = Path(__file__).with_name("synthetic_from_image.oifits")
    
    # Run the full MCMC analysis
    img_best, sparco_params, mcmc_params = fit_and_get_image(
        file_path=oifits_file,
        nx=128,
        pixsize=0.125,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_in=burn_in
    )
    
    return img_best, sparco_params, mcmc_params

def fit_and_get_image(file_path, nx=64, pixsize=0.125, n_walkers=32, n_steps=1000, burn_in=100):
    """
    Run full MCMC analysis and return best-fit image for reconstruction initialization.
    Does everything main() does but returns the image and params.
    
    Args:
        file_path: path to OIFITS file
        nx: image size in pixels
        pixsize: pixel scale (mas/pixel)
        n_walkers: number of MCMC walkers
        n_steps: number of MCMC steps
        burn_in: burn-in period
    
    Returns:
        img: best-fit disk image (nx x nx array)
        params: SPARCO parameters [f_star_0, f_bg_0, diameter, d_ind, lambda_0]
        best_params: MCMC best-fit parameters [inc, phi, radius, thickness, contrast, sigma, f_star_0]
    """
    print("="*80)
    print("MCMC Parametric Disk Fitting")
    print("="*80)
    print(f"Data file: {file_path}")
    print(f"MCMC config: {n_walkers} walkers, {n_steps} steps, burn-in={burn_in}")
    
    # Run MCMC
    print("\nRunning MCMC...")
    sampler, data, ft, fixed_params = run_mcmc_model(
        file_path=file_path,
        nx=nx,
        pixsize=pixsize,
        n_walkers=n_walkers,
        n_steps=n_steps
    )
    
    print("\nMCMC complete!")
    print(f"Chain shape: {sampler.get_chain().shape}")  # (nsteps, nwalkers, ndim)
    print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))
    
    # Get best fit parameters
    print("\nComputing best fit parameters...")
    best_params, median_params, std_params = get_best_fit(sampler, burn_in=burn_in)
    
    param_names = DISK_SPARCO_PARAM_NAMES
    
    # Compute chi2 for best fit
    print("\nComputing chi2 for best fit...")
    x = np.linspace(-0.5 * nx * pixsize, 0.5 * nx * pixsize, nx)
    y = np.linspace(-0.5 * nx * pixsize, 0.5 * nx * pixsize, nx)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    
    theta_best, sparco_params_best = unpack_params(best_params, fixed_params)
    img_best = render_disk(theta_best, x_grid, y_grid, apply_cutoff=True, normalize_flux=True)
    weights = (1.0, 0.0, 1.0)
    chi2_best = chi2_sparco_f(img_best, sparco_params_best, ft, data, verbose=False, weights=weights)
    
    # Calculate degrees of freedom
    n_vis2 = data.nv2
    n_t3phi = data.nt3phi
    n_data = n_vis2 + n_t3phi  # Only counting weighted data
    n_params = len(best_params)
    dof = n_data - n_params
    reduced_chi2 = chi2_best / dof
    
    print("\n" + "="*80)
    print("Best Fit Parameters (Maximum Log Probability)")
    print("="*80)
    print(f"Chi2: {chi2_best:.2f}")
    print(f"Degrees of Freedom: {dof} (n_data={n_data}, n_params={n_params})")
    print(f"Reduced Chi2: {reduced_chi2:.3f}")
    for name, val in zip(param_names, best_params):
        print(f"{name:20s}: {val:.6f}")
    
    print("\n" + "="*80)
    print("Median Parameters (with uncertainties)")
    print("="*80)
    for name, med, std in zip(param_names, median_params, std_params):
        print(f"{name:20s}: {med:.6f} ± {std:.6f}")
    
    # Plot chains (hairy caterpillar)
    print("\nCreating chain plot...")
    output_dir = Path(__file__).parent.parent / 'examples'
    output_dir.mkdir(exist_ok=True)
    chain_path = output_dir / 'mcmc_chains.png'
    
    fig_chains = plot_chains(sampler, save_path=chain_path)
    print(f"Saved chain plot to: {chain_path}")
    
    # Plot corner plot
    print("\nCreating corner plot...")
    corner_path = output_dir / 'mcmc_posterior_corner.png'
    
    fig = plot_posterior(sampler, burn_in=burn_in, thin=5, save_path=corner_path)
    print(f"Saved corner plot to: {corner_path}")
    
    # Plot disk images
    print("\nCreating disk image visualizations...")
    image_path = output_dir / 'mcmc_disk'
    fig_img, images = plot_disk_images(sampler, fixed_params, nx=nx, pixsize=pixsize, 
                                        burn_in=burn_in, save_path=str(image_path))
    # plt.show()  # Disabled to prevent image display during MCMC
    
    # Save individual images
    img_best_save, img_median, img_p16, img_p84, img_uncertainty = images
    np.savez(
        output_dir / 'mcmc_disk_images.npz',
        best=img_best_save,
        median=img_median,
        p16=img_p16,
        p84=img_p84,
        uncertainty=img_uncertainty
    )
    print(f"Saved disk images to: {output_dir / 'mcmc_disk_images.npz'}")

    # Save results
    results_path = output_dir / 'mcmc_results.npz'
    np.savez(
        results_path,
        chain=sampler.get_chain(),
        log_prob=sampler.get_log_prob(),
        best_params=best_params,
        median_params=median_params,
        std_params=std_params,
        param_names=param_names
    )
    print(f"Saved results to: {results_path}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    
    # Return image and params for reconstruction
    params = sparco_params_best  # [f_star_0, f_bg_0, diameter, d_ind, lambda_0]
    return img_best, params, best_params

if __name__ == '__main__':
    img, sparco_params, mcmc_params = main()
    
