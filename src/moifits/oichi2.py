"""
Chi-squared functions and NFFT setup for optical interferometry.
This code is adapted from the Julia OITOOLS package written by Fabien Baron, freely available on GitHub.
"""

import numpy as np
import os

class NFFTPlan:
    def __init__(self, uv_points, nx, pixsize_mas, backend="finufft", eps=1e-12, **backend_kwargs):
        # Julia flips u coordinate: scale_rad = ... * [-1;1] .* uv
        uv_points = np.asarray(uv_points, dtype=float)
        self.uv_points = np.array([- uv_points[0], uv_points[1]])
        self.nx = nx
        self.pixsize_mas = pixsize_mas
        self.backend = backend
        self.eps = eps
        self.backend_kwargs = backend_kwargs
        pix_rad = pixsize_mas * np.pi / 180.0 / 3.6e6
        coords = (np.arange(nx) - nx/2) * pix_rad
        xx, yy = np.meshgrid(coords, coords, indexing="ij")
        self.x = xx.ravel()
        self.y = yy.ravel()
        self.nthreads = self._choose_nthreads()
        self._gpu_modules = None

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
        """Transform image to complex visibilities using the configured backend."""
        if self.backend == "finufft":
            return self._forward_finufft(image)
        if self.backend == "cufinufft":
            return self._forward_cufinufft(image)
        if self.backend == "direct":
            return self._forward_direct(image)
        raise ValueError(f"Unsupported NFFT backend: {self.backend}")

    def adjoint(self, vis, real_output=True):
        """Adjoint transform from sampled visibilities to an image grid."""
        if self.backend == "finufft":
            return self._adjoint_finufft(vis, real_output=real_output)
        if self.backend == "cufinufft":
            return self._adjoint_cufinufft(vis, real_output=real_output)
        if self.backend == "direct":
            return self._adjoint_direct(vis, real_output=real_output)
        raise ValueError(f"Unsupported NFFT backend: {self.backend}")

    def _forward_finufft(self, image):
        import finufft

        u, v = self.uv_points  # already flipped in __init__
        I = np.asarray(image, np.complex128).ravel()
        vis = finufft.nufft2d3(
            self.x, self.y, I,
            2*np.pi*u, 2*np.pi*v,
            isign=-1, eps=self.eps, nthreads=self.nthreads
        )
        return vis

    def _adjoint_finufft(self, vis, real_output=True):
        import finufft

        u, v = self.uv_points
        img_flat = finufft.nufft2d3(
            2*np.pi*u, 2*np.pi*v,
            np.asarray(vis, np.complex128),
            self.x, self.y,
            isign=1, eps=self.eps, nthreads=self.nthreads
        )
        image = img_flat.reshape(self.nx, self.nx)
        return np.real(image) if real_output else image

    def _load_gpu_modules(self):
        if self._gpu_modules is None:
            try:
                import cupy as cp
                import cufinufft
            except ModuleNotFoundError as exc:
                missing = exc.name or "cupy/cufinufft"
                raise ModuleNotFoundError(
                    f"NFFT backend 'cufinufft' requires {missing}. "
                    "Install CUDA-compatible `cufinufft` and `cupy` packages."
                ) from exc
            device_id = self.backend_kwargs.get("gpu_device_id")
            if device_id is not None:
                cp.cuda.Device(int(device_id)).use()
            self._gpu_modules = cp, cufinufft
        return self._gpu_modules

    def _gpu_options(self):
        allowed = {
            "gpu_method",
            "gpu_sort",
            "gpu_kerevalmeth",
            "gpu_device_id",
            "gpu_stream",
            "modeord",
            "upsampfac",
        }
        return {key: value for key, value in self.backend_kwargs.items() if key in allowed}

    def _forward_cufinufft(self, image):
        cp, cufinufft = self._load_gpu_modules()
        u, v = self.uv_points
        vis_gpu = cufinufft.nufft2d3(
            cp.asarray(self.x),
            cp.asarray(self.y),
            cp.asarray(np.asarray(image, np.complex128).ravel()),
            cp.asarray(2*np.pi*u),
            cp.asarray(2*np.pi*v),
            isign=-1,
            eps=self.eps,
            **self._gpu_options(),
        )
        return cp.asnumpy(vis_gpu)

    def _adjoint_cufinufft(self, vis, real_output=True):
        cp, cufinufft = self._load_gpu_modules()
        u, v = self.uv_points
        image_gpu = cufinufft.nufft2d3(
            cp.asarray(2*np.pi*u),
            cp.asarray(2*np.pi*v),
            cp.asarray(np.asarray(vis, np.complex128)),
            cp.asarray(self.x),
            cp.asarray(self.y),
            isign=1,
            eps=self.eps,
            **self._gpu_options(),
        )
        image = cp.asnumpy(image_gpu).reshape(self.nx, self.nx)
        return np.real(image) if real_output else image

    def _forward_direct(self, image):
        u, v = self.uv_points
        image_flat = np.asarray(image, np.complex128).ravel()
        phase = 2*np.pi * (u[:, None] * self.x[None, :] + v[:, None] * self.y[None, :])
        return np.exp(-1j * phase) @ image_flat

    def _adjoint_direct(self, vis, real_output=True):
        u, v = self.uv_points
        phase = 2*np.pi * (u[:, None] * self.x[None, :] + v[:, None] * self.y[None, :])
        image = (np.asarray(vis, np.complex128) @ np.exp(1j * phase)).reshape(self.nx, self.nx)
        return np.real(image) if real_output else image

def setup_nfft(data, nx, pixsize, backend="finufft", eps=1e-12, **backend_kwargs):
    uv = data.uv      # already in λ^-1
    return [NFFTPlan(uv, nx, pixsize, backend=backend, eps=eps, **backend_kwargs),
            NFFTPlan(uv[:, data.indx_vis], nx, pixsize, backend=backend, eps=eps, **backend_kwargs),
            NFFTPlan(uv[:, data.indx_v2], nx, pixsize, backend=backend, eps=eps, **backend_kwargs),
            NFFTPlan(uv[:, data.indx_t3_1], nx, pixsize, backend=backend, eps=eps, **backend_kwargs),
            NFFTPlan(uv[:, data.indx_t3_2], nx, pixsize, backend=backend, eps=eps, **backend_kwargs),
            NFFTPlan(uv[:, data.indx_t3_3], nx, pixsize, backend=backend, eps=eps, **backend_kwargs)]


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
    return plan.adjoint(vis, real_output=real_output)


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
