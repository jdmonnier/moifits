"""
Chi-squared functions and NFFT setup for optical interferometry.
This code is adapted from the Julia OITOOLS package written by Fabien Baron, freely available on GitHub.
"""

import numpy as np
import finufft
import os
try:
    from .readoifits import OIData
except ImportError:  # Allow direct module import from scripts in moifits/testing
    from readoifits import OIData

class NFFTPlan:
    def __init__(self, uv_points, nx, pixsize_mas):
        # Julia flips u coordinate: scale_rad = ... * [-1;1] .* uv
        self.uv_points = np.array([- uv_points[0], uv_points[1]])
        self.nx = nx
        pix_rad = pixsize_mas * np.pi / 180.0 / 3.6e6
        coords = (np.arange(nx) - nx/2) * pix_rad
        xx, yy = np.meshgrid(coords, coords, indexing="ij")
        self.x = xx.ravel()
        self.y = yy.ravel()
        self.nthreads = self._choose_nthreads()

    def _choose_nthreads(self):
        env_nthreads = os.environ.get("FINUFFT_NTHREADS") or os.environ.get("OMP_NUM_THREADS")
        if env_nthreads:
            try:
                return max(1, int(env_nthreads))
            except ValueError:
                pass

        cpu_threads = os.cpu_count() or 1
        nsrc = self.x.size
        ntrg = self.uv_points.shape[1]
        workload = nsrc + ntrg
        if workload < 50_000:
            return 1
        if workload < 200_000:
            return min(cpu_threads, 4)
        return cpu_threads

    def forward(self, image):
        u, v = self.uv_points  # already flipped in __init__
        I = np.asarray(image, np.complex128).ravel()
        vis = finufft.nufft2d3(
            self.x, self.y, I,
            2*np.pi*u, 2*np.pi*v,
            isign=-1, eps=1e-12, nthreads=self.nthreads
        )
        return vis

def setup_nfft(data, nx, pixsize):
    uv = data.uv      # already in λ⁻¹
    return [NFFTPlan(uv, nx, pixsize),
            NFFTPlan(uv[:, data.indx_vis], nx, pixsize),
            NFFTPlan(uv[:, data.indx_v2], nx, pixsize),
            NFFTPlan(uv[:, data.indx_t3_1], nx, pixsize),
            NFFTPlan(uv[:, data.indx_t3_2], nx, pixsize),
            NFFTPlan(uv[:, data.indx_t3_3], nx, pixsize)]


def image_to_vis(x, nfft_plan):
    """Transform image to complex visibilities using NFFT."""
    if isinstance(nfft_plan, list):
        nfft_plan = nfft_plan[0]

    if x.ndim == 1:
        nx = int(np.sqrt(len(x)))
        x = x.reshape(nx, nx)
    x_norm = x / np.sum(x)

    return nfft_plan.forward(x_norm)


def vis_to_v2(cvis, indx):
    """Convert complex visibilities to squared visibilities."""
    return np.abs(cvis[indx])**2


def vis_to_t3(cvis, indx1, indx2, indx3):
    """Compute closure phases from complex visibilities."""
    t3 = cvis[indx1] * cvis[indx2] * cvis[indx3]
    t3amp = np.abs(t3)
    t3phi = np.angle(t3) * 180.0 / np.pi
    return t3, t3amp, t3phi


def image_to_obs(x, ft, data):
    """Transform image to all observables (V2, T3amp, T3phi)."""
    cvis_model = image_to_vis(x, ft)
    v2_model = vis_to_v2(cvis_model, data.indx_v2)
    _, t3amp_model, t3phi_model = vis_to_t3(cvis_model,
                                            data.indx_t3_1,
                                            data.indx_t3_2,
                                            data.indx_t3_3)
    return v2_model, t3amp_model, t3phi_model


def mod360(x):
    """Wrap angles to [-180, 180] range."""
    return ((x + 180.0) % 360.0) - 180.0


def chi2_nfft(x, ftplan, data, weights=[1.0, 1.0, 1.0], verbose=False):
    """Compute chi-squared for interferometric data using NFFT."""
    flux = np.sum(x)
    cvis_model = image_to_vis(x, ftplan[0])

    v2_model = vis_to_v2(cvis_model, data.indx_v2)
    _, t3amp_model, t3phi_model = vis_to_t3(cvis_model,
                                            data.indx_t3_1,
                                            data.indx_t3_2,
                                            data.indx_t3_3)

    chi2_v2 = chi2_t3amp = chi2_t3phi = 0.0

    if weights[0] > 0 and data.nv2 > 0:
        chi2_v2 = np.sum(((v2_model - data.v2) / data.v2_err)**2)

    if weights[1] > 0 and data.nt3amp > 0:
        chi2_t3amp = np.sum(((t3amp_model - data.t3amp) / data.t3amp_err)**2)

    if weights[2] > 0 and data.nt3phi > 0:
        chi2_t3phi = np.sum((mod360(t3phi_model - data.t3phi) / data.t3phi_err)**2)

    if verbose:
        print(f"V2: {chi2_v2/data.nv2:.4f} ", end='')
        print(f"T3A: {chi2_t3amp/data.nt3amp:.4f} ", end='')
        print(f"T3P: {chi2_t3phi/data.nt3phi:.4f} ", end='')
        print(f"Flux: {flux:.4f}")

    return weights[0]*chi2_v2 + weights[1]*chi2_t3amp + weights[2]*chi2_t3phi


def nfft_adjoint(plan, vis, real_output=True):
    """
    Adjoint NFFT: transforms visibilities back to image plane.
    
    The forward transform was:
      vis = nufft2d3(x_spatial, y_spatial, I, 2π*u, 2π*v, isign=-1)
      
    The adjoint is:
      I_adjoint = nufft2d3(2π*u, 2π*v, vis_conj, x_spatial, y_spatial, isign=+1)
      
    This swaps the role of spatial and frequency coordinates.
    """
    u, v = plan.uv_points
    img_flat = finufft.nufft2d3(
        2*np.pi*u, 2*np.pi*v,
        vis.astype(np.complex128),
        plan.x, plan.y,
        isign=1, eps=1e-12, nthreads=plan.nthreads
    )
    return np.real(img_flat.reshape(plan.nx, plan.nx)) if real_output else img_flat.reshape(plan.nx, plan.nx)


def chi2_fg(x, g, ftplan, data, weights=[1.0, 1.0, 1.0], verbose=False, vonmises=False):
    """
    Compute chi-squared and its gradient for interferometric data using NFFT.
    
    Args:
        x: 2D image array (nx, nx)
        g: 2D gradient array (nx, nx) - modified in place
        ftplan: List of NFFTPlan objects [uv, vis, v2, t3_1, t3_2, t3_3]
        data: OIData object
        weights: [w_v2, w_t3amp, w_t3phi]
        verbose: Print chi2 components
        vonmises: Use von Mises distribution for closure phases
        
    Returns:
        chi2: Total chi-squared value
    """
    flux = np.sum(x)
    cvis_model = image_to_vis(x, ftplan[0])
    
    v2_model = vis_to_v2(cvis_model, data.indx_v2)
    t3_model, t3amp_model, t3phi_model = vis_to_t3(cvis_model,
                                                     data.indx_t3_1,
                                                     data.indx_t3_2,
                                                     data.indx_t3_3)
    
    chi2_v2 = chi2_t3amp = chi2_t3phi = 0.0
    g_v2 = g_t3amp = g_t3phi = 0.0
    
    # V2 gradient
    if weights[0] > 0 and data.nv2 > 0:
        chi2_v2 = np.sum(((v2_model - data.v2) / data.v2_err)**2)
        dchi2_dv2 = 4.0 * ((v2_model - data.v2) / data.v2_err**2) * cvis_model[data.indx_v2]
        g_v2 = nfft_adjoint(ftplan[2], dchi2_dv2)
    
    # T3amp gradient
    if weights[1] > 0 and data.nt3amp > 0:
        chi2_t3amp = np.sum(((t3amp_model - data.t3amp) / data.t3amp_err)**2)
        dT3 = 2.0 * (t3amp_model - data.t3amp) / data.t3amp_err**2
        
        # Gradient contributions from each baseline in the triangle
        g1 = dT3 * cvis_model[data.indx_t3_1] / np.abs(cvis_model[data.indx_t3_1]) * \
             np.abs(cvis_model[data.indx_t3_2]) * np.abs(cvis_model[data.indx_t3_3])
        g2 = dT3 * cvis_model[data.indx_t3_2] / np.abs(cvis_model[data.indx_t3_2]) * \
             np.abs(cvis_model[data.indx_t3_1]) * np.abs(cvis_model[data.indx_t3_3])
        g3 = dT3 * cvis_model[data.indx_t3_3] / np.abs(cvis_model[data.indx_t3_3]) * \
             np.abs(cvis_model[data.indx_t3_1]) * np.abs(cvis_model[data.indx_t3_2])
        
        g_t3amp = (nfft_adjoint(ftplan[3], g1) + 
                   nfft_adjoint(ftplan[4], g2) + 
                   nfft_adjoint(ftplan[5], g3))
    
    # T3phi gradient
    if weights[2] > 0 and data.nt3phi > 0:
        if not vonmises:
            chi2_t3phi = np.sum((mod360(t3phi_model - data.t3phi) / data.t3phi_err)**2)
            dT3 = -360.0 / np.pi * mod360(t3phi_model - data.t3phi) / data.t3phi_err**2
            
            # Gradient contributions (imaginary part)
            g1 = dT3 / np.abs(cvis_model[data.indx_t3_1])**2 * cvis_model[data.indx_t3_1]
            g2 = dT3 / np.abs(cvis_model[data.indx_t3_2])**2 * cvis_model[data.indx_t3_2]
            g3 = dT3 / np.abs(cvis_model[data.indx_t3_3])**2 * cvis_model[data.indx_t3_3]

            g_t3phi = (np.imag(nfft_adjoint(ftplan[3], g1, real_output=False)) +
                       np.imag(nfft_adjoint(ftplan[4], g2, real_output=False)) +
                       np.imag(nfft_adjoint(ftplan[5], g3, real_output=False)))
        else:
            # von Mises distribution
            chi2_t3phi = np.sum(-2 * data.t3phi_vonmises_err * 
                               np.cos((t3phi_model - data.t3phi) / 180.0 * np.pi) + 
                               data.t3phi_vonmises_chi2_offset)
            dT3 = -2.0 * data.t3phi_vonmises_err * \
                  np.sin((t3phi_model - data.t3phi) / 180.0 * np.pi)
            
            g1 = dT3 / np.abs(cvis_model[data.indx_t3_1])**2 * cvis_model[data.indx_t3_1]
            g2 = dT3 / np.abs(cvis_model[data.indx_t3_2])**2 * cvis_model[data.indx_t3_2]
            g3 = dT3 / np.abs(cvis_model[data.indx_t3_3])**2 * cvis_model[data.indx_t3_3]
            
            g_t3phi = (np.imag(nfft_adjoint(ftplan[3], g1)) + 
                       np.imag(nfft_adjoint(ftplan[4], g2)) + 
                       np.imag(nfft_adjoint(ftplan[5], g3)))
    
    # Combine gradients
    g[:] = weights[0] * g_v2 + weights[1] * g_t3amp + weights[2] * g_t3phi
    
    if verbose:
        if weights[0] > 0 and data.nv2 > 0:
            print(f"V2: {chi2_v2/data.nv2:.4f} ", end='')
        print(f"T3A: {chi2_t3amp/data.nt3amp:.4f} ", end='')
        print(f"T3P: {chi2_t3phi/data.nt3phi:.4f} ", end='')
        print(f"Flux: {flux:.4f}")

    print("||g_v2||", np.linalg.norm(g_v2),
      "||g_t3amp||", np.linalg.norm(g_t3amp),
      "||g_t3phi||", np.linalg.norm(g_t3phi))
    
    return weights[0]*chi2_v2 + weights[1]*chi2_t3amp + weights[2]*chi2_t3phi
