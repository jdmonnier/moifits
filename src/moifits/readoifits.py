'''
This code is adapted from the Julia OITOOLS package written by Fabien Baron, freely available on GitHub.
'''

import numpy as np
import astropy.io.fits as fits
from dataclasses import dataclass, field
from typing import List, Tuple
import matplotlib.colors as mcolors
from scipy.spatial import KDTree

@dataclass
class OIData:
    """
    Container for optical interferometry data from OIFITS files.
    """
    # Complex visibilities
    visamp: np.ndarray = field(default_factory=lambda: np.array([]))
    visamp_err: np.ndarray = field(default_factory=lambda: np.array([]))
    visphi: np.ndarray = field(default_factory=lambda: np.array([]))
    visphi_err: np.ndarray = field(default_factory=lambda: np.array([]))
    vis_baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    vis_mjd: np.ndarray = field(default_factory=lambda: np.array([]))
    vis_lam: np.ndarray = field(default_factory=lambda: np.array([]))
    vis_dlam: np.ndarray = field(default_factory=lambda: np.array([]))
    vis_flag: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    
    # V2 (squared visibilities)
    v2: np.ndarray = field(default_factory=lambda: np.array([]))
    v2_err: np.ndarray = field(default_factory=lambda: np.array([]))
    v2_baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    v2_mjd: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_mjd: float = 0.0
    v2_lam: np.ndarray = field(default_factory=lambda: np.array([]))
    v2_dlam: np.ndarray = field(default_factory=lambda: np.array([]))
    v2_flag: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    
    # T3 (closure phases and triple amplitudes)
    t3amp: np.ndarray = field(default_factory=lambda: np.array([]))
    t3amp_err: np.ndarray = field(default_factory=lambda: np.array([]))
    t3phi: np.ndarray = field(default_factory=lambda: np.array([]))
    t3phi_err: np.ndarray = field(default_factory=lambda: np.array([]))
    t3phi_vonmises_err: np.ndarray = field(default_factory=lambda: np.array([]))
    t3phi_vonmises_chi2_offset: np.ndarray = field(default_factory=lambda: np.array([]))
    t3_baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    t3_maxbaseline: np.ndarray = field(default_factory=lambda: np.array([]))
    t3_mjd: np.ndarray = field(default_factory=lambda: np.array([]))
    t3_lam: np.ndarray = field(default_factory=lambda: np.array([]))
    t3_dlam: np.ndarray = field(default_factory=lambda: np.array([]))
    t3_flag: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    
    # OIFlux
    flux: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_err: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_mjd: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_lam: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_dlam: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_flag: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    flux_sta_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    
    # UV coverage
    uv: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))  # 2xN array
    uv_lam: np.ndarray = field(default_factory=lambda: np.array([]))
    uv_dlam: np.ndarray = field(default_factory=lambda: np.array([]))
    uv_mjd: np.ndarray = field(default_factory=lambda: np.array([]))
    uv_baseline: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Data product sizes
    nflux: int = 0
    nvisamp: int = 0
    nvisphi: int = 0
    nv2: int = 0
    nt3amp: int = 0
    nt3phi: int = 0
    nuv: int = 0
    
    # Indexing logic (1-based in Julia, 0-based in Python)
    indx_vis: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    indx_v2: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    indx_t3_1: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    indx_t3_2: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    indx_t3_3: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    
    # Station/telescope information
    sta_name: List[str] = field(default_factory=list)
    tel_name: List[str] = field(default_factory=list)
    sta_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    vis_sta_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), dtype=int))
    v2_sta_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), dtype=int))
    t3_sta_index: np.ndarray = field(default_factory=lambda: np.zeros((3, 0), dtype=int))
    
    # Filename
    filename: str = ""
    
    def __repr__(self):
        """Pretty print data summary"""
        lines = [
            f"OIData from: {self.filename}",
            f"Mean MJD: {self.mean_mjd:.2f}",
            f"Wavelength range: {self.uv_lam.min():.2e} - {self.uv_lam.max():.2e} m" if len(self.uv_lam) > 0 else "No wavelength data",
            f"nflux: {self.nflux} | nuv: {self.nuv} | nvisamp: {self.nvisamp} | nvisphi: {self.nvisphi}",
            f"nv2: {self.nv2} | nt3amp: {self.nt3amp} | nt3phi: {self.nt3phi}"
        ]
        return "\n".join(lines)


def display_oidata(data: OIData):
    """Display summary of a single OIData object"""
    print(f"Mean MJD: {data.mean_mjd}")
    if len(data.uv_lam) > 0:
        print(f"Wavelength range: {data.uv_lam.min()} - {data.uv_lam.max()}")
    else:
        print("Wavelength range: No data")
    print(f"nflux: {data.nflux} | nuv: {data.nuv} | nvisamp: {data.nvisamp} | "
          f"nvisphi: {data.nvisphi} | nv2: {data.nv2} | nt3amp: {data.nt3amp} | nt3phi: {data.nt3phi}")


def display_oidata_array(data_array: np.ndarray):
    """
    Display summary of an array of OIData objects (wavelength x time bins)
    
    Args:
        data_array: 2D numpy array of OIData objects [nwav, ntime]
    """
    nwav, ntime = data_array.shape
    print(f"Original data file: {data_array[0, 0].filename}")
    print(f"Number of wavelength bins: {nwav}")
    print(f"Number of time/epoch bins: {ntime}")
    print()
    
    if nwav == 1 and ntime == 1:
        display_oidata(data_array[0, 0])
    elif ntime == 1:
        # Use matplotlib's tab10 colormap for up to 10 wavelengths
        cmap = mcolors.TABLEAU_COLORS
        color_list = list(cmap.values())
        
        for i in range(nwav):
            # Convert matplotlib hex color to ANSI
            color_hex = color_list[i % len(color_list)]
            rgb = mcolors.hex2color(color_hex)
            r, g, b = [int(c * 255) for c in rgb]
            ansi_color = f'\033[38;2;{r};{g};{b}m'
            reset = '\033[0m'
            bold = '\033[1m'
            
            print(f"{bold}{ansi_color}Wavelength bin {i+1}/{nwav}{reset}")
            display_oidata(data_array[i, 0])
            print()


def set_data_filter(data: OIData, 
                   wav_range: List[List[float]] = None,
                   mjd_range: List[List[float]] = None, 
                   baseline_range: List[float] = None,
                   filter_bad_data: bool = False,
                   filter_vis: bool = True,
                   filter_v2: bool = True, 
                   filter_t3amp: bool = True,
                   filter_t3phi: bool = True,
                   cutoff_minv2: float = -1.0,
                   cutoff_maxv2: float = 2.0,
                   cutoff_mint3amp: float = -1.0,
                   cutoff_maxt3amp: float = 1.5,
                   special_filter_diffvis: bool = False,
                   force_full_vis: bool = False,
                   force_full_t3: bool = False,
                   filter_v2_snr_threshold: float = 0.01,
                   uv_bad: List[int] = None,
                   filter_visphi: bool = False,
                   filter_visamp: bool = False):
    """
    Identify which data points should be discarded based on filtering criteria.
    
    Returns:
        List of [uv_bad, vis_bad, v2_bad, t3_bad] index arrays (0-based)
    """
    # Handle default arguments
    if wav_range is None:
        wav_range = [[-1.0, 1e99]]
    elif isinstance(wav_range[0], (int, float)):
        wav_range = [wav_range]
        
    if mjd_range is None:
        mjd_range = [[-1.0, 1e99]]
    elif isinstance(mjd_range[0], (int, float)):
        mjd_range = [mjd_range]
        
    if baseline_range is None:
        baseline_range = [0, 1e99]
        
    if uv_bad is None:
        uv_bad = []
    else:
        uv_bad = list(uv_bad)
    
    # Determine which data types to use
    use_visphi = (data.nvisphi > 0) and (filter_vis or filter_visphi)
    use_visamp = (data.nvisamp > 0) and (filter_vis or filter_visamp)
    use_vis = use_visphi or use_visamp
    use_v2 = (data.nv2 > 0) and filter_v2
    use_t3amp = (data.nt3amp > 0) and filter_t3amp
    use_t3phi = (data.nt3phi > 0) and filter_t3phi
    use_t3 = use_t3phi or use_t3amp
    
    # Track bad data points
    vis_bad = []
    v2_bad = []
    t3_bad = []
    
    # Filter obviously bad data
    if filter_bad_data:
        if use_vis:
            visamp_good = (~np.isnan(data.visamp)) & (~np.isnan(data.visamp_err)) & (data.visamp_err > 0.0)
            visphi_good = (~np.isnan(data.visphi)) & (~np.isnan(data.visphi_err)) & (data.visphi_err > 0.0)
            
            if not force_full_vis:
                vis_good = np.where(~data.vis_flag & (visamp_good | visphi_good))[0]
            else:
                vis_good = np.where(~data.vis_flag & (visamp_good & visphi_good))[0]
            
            if special_filter_diffvis:
                vis_good = np.where(data.vis_flag != 2)[0]
            
            vis_bad = list(set(range(len(data.vis_flag))) - set(vis_good))
        
        if use_v2:
            # Filter bad V2: flagged, bad errors, out of range, low SNR
            v2_good = np.where(
                (~data.v2_flag) & 
                (data.v2_err > 0.0) & (data.v2_err < 1.0) &
                (data.v2 > cutoff_minv2) & (data.v2 < cutoff_maxv2) &
                (~np.isnan(data.v2)) & (~np.isnan(data.v2_err)) &
                (np.abs(data.v2 / data.v2_err) > filter_v2_snr_threshold)
            )[0]
            v2_bad = list(set(range(len(data.v2_flag))) - set(v2_good))
        
        if use_t3:
            t3amp_good = (~np.isnan(data.t3amp)) & (~np.isnan(data.t3amp_err)) & (data.t3amp_err > 0.0)
            t3phi_good = (~np.isnan(data.t3phi)) & (~np.isnan(data.t3phi_err)) & (data.t3phi_err > 0.0)
            
            if not force_full_t3:
                t3_good = np.where(~data.t3_flag & (t3amp_good | t3phi_good))[0]
            else:
                t3_good = np.where(
                    ~data.t3_flag & (t3amp_good & t3phi_good) &
                    (data.t3amp > cutoff_mint3amp) & (data.t3amp < cutoff_maxt3amp)
                )[0]
            
            t3_bad = list(set(range(len(data.t3_flag))) - set(t3_good))
    
    # Filter UV plane (this cascades to observables)
    if baseline_range != [0, 1e99]:
        uv_bad.extend(np.where(data.uv_baseline < baseline_range[0])[0].tolist())
        uv_bad.extend(np.where(data.uv_baseline > baseline_range[1])[0].tolist())
    
    uv_good = set(range(data.nuv)) - set(uv_bad)
    
    # Filter by MJD ranges
    if mjd_range != [[-1.0, 1e99]]:
        mjd_good = []
        for mjd_min, mjd_max in mjd_range:
            mjd_good.extend(np.where((data.uv_mjd >= mjd_min) & (data.uv_mjd <= mjd_max))[0].tolist())
        uv_good = uv_good & set(mjd_good)
    
    # Filter by wavelength ranges
    if wav_range != [[-1.0, 1e99]]:
        wav_good = []
        for wav_min, wav_max in wav_range:
            wav_good.extend(np.where((data.uv_lam >= wav_min) & (data.uv_lam <= wav_max))[0].tolist())
        uv_good = uv_good & set(wav_good)
    
    uv_bad = list(set(range(data.nuv)) - uv_good)
    
    # Mark observables as bad if their UV points are bad
    if data.nvisamp > 0 or data.nvisphi > 0:
        vis_bad.extend([i for i in range(len(data.indx_vis)) if data.indx_vis[i] not in uv_good])
    
    if data.nv2 > 0:
        v2_bad.extend([i for i in range(len(data.indx_v2)) if data.indx_v2[i] not in uv_good])
    
    if data.nt3amp > 0 or data.nt3phi > 0:
        t3_bad.extend([i for i in range(len(data.indx_t3_1)) 
                      if (data.indx_t3_1[i] not in uv_good or 
                          data.indx_t3_2[i] not in uv_good or 
                          data.indx_t3_3[i] not in uv_good)])
    
    return [uv_bad, vis_bad, v2_bad, t3_bad]


def filter_data(data_in: OIData, indexes_to_discard: List[List[int]]) -> OIData:
    """
    Apply filtering by removing bad data points and rebuilding UV coverage.
    
    Args:
        data_in: Input OIData object
        indexes_to_discard: [uv_bad, vis_bad, v2_bad, t3_bad] from set_data_filter
        
    Returns:
        New filtered OIData object
    """
    # Deep copy to avoid modifying input
    import copy
    data = copy.deepcopy(data_in)
    
    good_uv_vis = []
    good_uv_v2 = []
    good_uv_t3_1 = []
    good_uv_t3_2 = []
    good_uv_t3_3 = []
    
    # Filter complex visibilities
    if data.nvisamp > 0 or data.nvisphi > 0:
        vis_good = list(set(range(len(data.indx_vis))) - set(indexes_to_discard[1]))
        vis_good.sort()
        good_uv_vis = data.indx_vis[vis_good]
        data.visamp = data.visamp[vis_good]
        data.visamp_err = data.visamp_err[vis_good]
        data.visphi = data.visphi[vis_good]
        data.visphi_err = data.visphi_err[vis_good]
        data.vis_baseline = data.vis_baseline[vis_good]
        data.vis_mjd = data.vis_mjd[vis_good]
        data.vis_lam = data.vis_lam[vis_good]
        data.vis_dlam = data.vis_dlam[vis_good]
        data.vis_flag = data.vis_flag[vis_good]
        data.vis_sta_index = data.vis_sta_index[:, vis_good]
        data.nvisamp = len(data.visamp)
        data.nvisphi = len(data.visphi)
    
    # Filter V2
    if data.nv2 > 0:
        v2_good = list(set(range(len(data.indx_v2))) - set(indexes_to_discard[2]))
        v2_good.sort()
        good_uv_v2 = data.indx_v2[v2_good]
        data.v2 = data.v2[v2_good]
        data.v2_err = data.v2_err[v2_good]
        data.v2_baseline = data.v2_baseline[v2_good]
        data.v2_mjd = data.v2_mjd[v2_good]
        data.v2_lam = data.v2_lam[v2_good]
        data.v2_dlam = data.v2_dlam[v2_good]
        data.v2_flag = data.v2_flag[v2_good]
        data.v2_sta_index = data.v2_sta_index[:, v2_good]
        data.nv2 = len(data.v2)
    
    # Filter T3
    if data.nt3amp > 0 or data.nt3phi > 0:
        t3_good = list(set(range(len(data.indx_t3_1))) - set(indexes_to_discard[3]))
        t3_good.sort()
        good_uv_t3_1 = data.indx_t3_1[t3_good]
        good_uv_t3_2 = data.indx_t3_2[t3_good]
        good_uv_t3_3 = data.indx_t3_3[t3_good]
        data.t3amp = data.t3amp[t3_good]
        data.t3amp_err = data.t3amp_err[t3_good]
        data.t3phi = data.t3phi[t3_good]
        data.t3phi_err = data.t3phi_err[t3_good]
        data.t3_baseline = data.t3_baseline[t3_good]
        data.t3_maxbaseline = data.t3_maxbaseline[t3_good]
        data.t3_mjd = data.t3_mjd[t3_good]
        data.t3_lam = data.t3_lam[t3_good]
        data.t3_dlam = data.t3_dlam[t3_good]
        data.t3_flag = data.t3_flag[t3_good]
        data.t3_sta_index = data.t3_sta_index[:, t3_good]
        data.nt3amp = len(data.t3amp)
        data.nt3phi = len(data.t3phi)
    
    # Rebuild UV coverage with only used points
    uv_select = np.zeros(data.nuv, dtype=bool)
    
    if len(good_uv_vis) > 0:
        uv_select[good_uv_vis] = True
    if len(good_uv_v2) > 0:
        uv_select[good_uv_v2] = True
    if len(good_uv_t3_1) > 0:
        uv_select[good_uv_t3_1] = True
        uv_select[good_uv_t3_2] = True
        uv_select[good_uv_t3_3] = True
    
    # Disable explicitly bad UV points
    for bad_idx in indexes_to_discard[0]:
        uv_select[bad_idx] = False
    
    # Build index conversion map (old UV indices -> new compressed indices)
    indx_conv = np.zeros(len(uv_select), dtype=int)
    acc = 0
    for i in range(len(uv_select)):
        if uv_select[i]:
            acc += 1
        indx_conv[i] = acc - 1  # 0-based indexing
    
    # Apply UV filtering
    indx_uv_sel = np.where(uv_select)[0]
    data.uv = data.uv[:, indx_uv_sel]
    data.uv_lam = data.uv_lam[indx_uv_sel]
    data.uv_dlam = data.uv_dlam[indx_uv_sel]
    data.uv_mjd = data.uv_mjd[indx_uv_sel]
    data.uv_baseline = data.uv_baseline[indx_uv_sel]
    data.nuv = data.uv.shape[1]
    
    # Remap observable indices to new compressed UV array
    if len(good_uv_vis) > 0:
        data.indx_vis = indx_conv[good_uv_vis]
    if len(good_uv_v2) > 0:
        data.indx_v2 = indx_conv[good_uv_v2]
    if len(good_uv_t3_1) > 0:
        data.indx_t3_1 = indx_conv[good_uv_t3_1]
        data.indx_t3_2 = indx_conv[good_uv_t3_2]
        data.indx_t3_3 = indx_conv[good_uv_t3_3]
    
    return data


def rm_redundance_kdtree(uv: np.ndarray, uvtol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find redundant UV points using KDTree spatial search.
    
    Matches Julia's OITOOLS implementation exactly:
    - For each point, find all points within uvtol
    - Assign all those points to the minimum index in the group
    - This creates transitive merging
    
    Args:
        uv: 2xN array of UV coordinates
        uvtol: Distance tolerance for considering points redundant
        
    Returns:
        indx_red_conv: Conversion array mapping old indices to new kept indices (0-based)
        to_keep: Array of unique UV point indices to keep (0-based)
    """
    nuv = uv.shape[1]
    
    if nuv == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Initialize: each point maps to itself (0-based indexing)
    indx_redundance = np.arange(nuv)
    
    # Build KDTree for spatial queries
    kdtree = KDTree(uv.T)  # Transpose to (N, 2) for KDTree
    
    # For each point, find nearby points and map them to the minimum index
    for i in range(nuv):
        # Query points within tolerance (returns indices)
        redundant_indices = kdtree.query_ball_point(uv[:, i], uvtol)
        
        if len(redundant_indices) > 0:
            # All redundant points map to the minimum index in the group
            min_idx = min(redundant_indices)
            for idx in redundant_indices:
                indx_redundance[idx] = min_idx
    
    # Get unique indices to keep (these are the "representatives" of each group)
    to_keep = np.unique(indx_redundance)
    
    # Build conversion map: old index -> new compressed index
    # This is equivalent to Julia's: indexin(indx_redundance, tokeep)
    # which finds the position of each indx_redundance[i] value in tokeep
    indx_red_conv = np.searchsorted(to_keep, indx_redundance)
    
    return indx_red_conv, to_keep


def remove_redundant_uv(data_in: OIData, uvtol: float = 2e2) -> OIData:
    """
    Remove redundant UV points and reassign observable indices.
    
    Args:
        data_in: Input OIData object
        uvtol: Distance tolerance for merging UV points (default: 200)
        
    Returns:
        New OIData with redundant UV points removed
    """
    import copy
    data = copy.deepcopy(data_in)
    
    # Find redundancies
    indx_red_conv, to_keep = rm_redundance_kdtree(data.uv, uvtol)
    
    # Filter UV arrays
    data.uv = data.uv[:, to_keep]
    data.uv_lam = data.uv_lam[to_keep]
    data.uv_dlam = data.uv_dlam[to_keep]
    data.uv_mjd = data.uv_mjd[to_keep]
    data.uv_baseline = data.uv_baseline[to_keep]
    data.nuv = len(to_keep)
    
    # Remap observable indices to merged UV points
    if data.nvisphi > 0 or data.nvisamp > 0:
        data.indx_vis = indx_red_conv[data.indx_vis]
    
    if data.nv2 > 0:
        data.indx_v2 = indx_red_conv[data.indx_v2]
    
    if data.nt3amp > 0 or data.nt3phi > 0:
        data.indx_t3_1 = indx_red_conv[data.indx_t3_1]
        data.indx_t3_2 = indx_red_conv[data.indx_t3_2]
        data.indx_t3_3 = indx_red_conv[data.indx_t3_3]
    
    return data


# =============================================================================
# OIFITS FILE PARSING
# =============================================================================

def _extract_wavelength_info(hdul):
    """Extract wavelength information from OI_WAVELENGTH table(s)."""
    wav_tables = [hdu for hdu in hdul if hdu.name == 'OI_WAVELENGTH']
    if not wav_tables:
        raise ValueError("No OI_WAVELENGTH table found in OIFITS file")
    
    # Build mapping: INSNAME -> (eff_wave, eff_band)
    wav_dict = {}
    for table in wav_tables:
        insname = table.header.get('INSNAME', '')
        wav_dict[insname] = {
            'eff_wave': np.array(table.data['EFF_WAVE']),
            'eff_band': np.array(table.data['EFF_BAND'])
        }
    return wav_dict


def _parse_oi_vis(hdul, wav_dict, target_filter=None):
    """Parse OI_VIS tables (complex visibilities)."""
    vis_tables = [hdu for hdu in hdul if hdu.name == 'OI_VIS']
    if not vis_tables:
        return None
    
    all_visamp = []
    all_visamp_err = []
    all_visphi = []
    all_visphi_err = []
    all_vis_ucoord = []
    all_vis_vcoord = []
    all_vis_mjd = []
    all_vis_lam = []
    all_vis_dlam = []
    all_vis_flag = []
    all_vis_sta_index = []
    
    for table in vis_tables:
        data = table.data
        insname = table.header.get('INSNAME', '')
        
        # Filter by target if requested
        if target_filter is not None:
            mask = np.isin(data['TARGET_ID'], target_filter)
        else:
            mask = np.ones(len(data), dtype=bool)
        
        # Get wavelength info
        wav_info = wav_dict.get(insname, {})
        eff_wave = wav_info.get('eff_wave', np.array([1.0]))
        eff_band = wav_info.get('eff_band', np.array([0.0]))
        
        # Extract data - each row is an observation with an array of wavelengths
        visamp = np.array([row for row in data['VISAMP'][mask]])  # shape: (nobs, nwave)
        visamperr = np.array([row for row in data['VISAMPERR'][mask]])
        visphi = np.array([row for row in data['VISPHI'][mask]])
        visphierr = np.array([row for row in data['VISPHIERR'][mask]])
        ucoord = data['UCOORD'][mask]
        vcoord = data['VCOORD'][mask]
        mjd = data['MJD'][mask]
        flag = np.array([row for row in data['FLAG'][mask]])  # shape: (nobs, nwave)
        sta_index = np.array([row for row in data['STA_INDEX'][mask]]).T  # shape: (2, nobs)
        
        # Flatten to per-datapoint arrays
        nobs, nwave = visamp.shape
        all_visamp.append(visamp.ravel())
        all_visamp_err.append(visamperr.ravel())
        all_visphi.append(visphi.ravel())
        all_visphi_err.append(visphierr.ravel())
        all_vis_flag.append(flag.ravel())
        
        # Repeat per-observation metadata for each wavelength
        all_vis_ucoord.extend(np.repeat(ucoord, nwave))
        all_vis_vcoord.extend(np.repeat(vcoord, nwave))
        all_vis_mjd.extend(np.repeat(mjd, nwave))
        all_vis_lam.extend(np.tile(eff_wave, nobs))
        all_vis_dlam.extend(np.tile(eff_band, nobs))
        
        # Station indices (2 x nobs*nwave)
        sta_repeated = np.repeat(sta_index, nwave, axis=1)
        all_vis_sta_index.append(sta_repeated)
    
    if not all_visamp:
        return None
    
    return {
        'visamp': np.concatenate(all_visamp),
        'visamp_err': np.concatenate(all_visamp_err),
        'visphi': np.concatenate(all_visphi),
        'visphi_err': np.concatenate(all_visphi_err),
        'vis_ucoord': np.array(all_vis_ucoord),
        'vis_vcoord': np.array(all_vis_vcoord),
        'vis_mjd': np.array(all_vis_mjd),
        'vis_lam': np.array(all_vis_lam),
        'vis_dlam': np.array(all_vis_dlam),
        'vis_flag': np.concatenate(all_vis_flag),
        'vis_sta_index': np.hstack(all_vis_sta_index) if all_vis_sta_index else np.zeros((2, 0), dtype=int)
    }


def _parse_oi_vis2(hdul, wav_dict, target_filter=None):
    """Parse OI_VIS2 tables (squared visibilities)."""
    v2_tables = [hdu for hdu in hdul if hdu.name == 'OI_VIS2']
    if not v2_tables:
        return None
    
    all_v2 = []
    all_v2_err = []
    all_v2_ucoord = []
    all_v2_vcoord = []
    all_v2_mjd = []
    all_v2_lam = []
    all_v2_dlam = []
    all_v2_flag = []
    all_v2_sta_index = []
    
    for table in v2_tables:
        data = table.data
        insname = table.header.get('INSNAME', '')
        
        # Filter by target if requested
        if target_filter is not None:
            mask = np.isin(data['TARGET_ID'], target_filter)
        else:
            mask = np.ones(len(data), dtype=bool)
        
        # Get wavelength info
        wav_info = wav_dict.get(insname, {})
        eff_wave = wav_info.get('eff_wave', np.array([1.0]))
        eff_band = wav_info.get('eff_band', np.array([0.0]))
        
        # Extract data - each row is an observation with an array of wavelengths
        # Apply mask first to select rows, then stack into 2D array
        vis2data = np.array([row for row in data['VIS2DATA'][mask]])  # shape: (nobs, nwave)
        vis2err = np.array([row for row in data['VIS2ERR'][mask]])
        ucoord = data['UCOORD'][mask]
        vcoord = data['VCOORD'][mask]
        mjd = data['MJD'][mask]
        flag = np.array([row for row in data['FLAG'][mask]])  # shape: (nobs, nwave)
        sta_index = np.array([row for row in data['STA_INDEX'][mask]]).T  # shape: (2, nobs)
        
        # Flatten to per-datapoint arrays
        nobs, nwave = vis2data.shape
        all_v2.append(vis2data.ravel())  # Already (nobs, nwave), just ravel
        all_v2_err.append(vis2err.ravel())
        all_v2_flag.append(flag.ravel())
        
        # Repeat per-observation metadata for each wavelength
        all_v2_ucoord.extend(np.repeat(ucoord, nwave))
        all_v2_vcoord.extend(np.repeat(vcoord, nwave))
        all_v2_mjd.extend(np.repeat(mjd, nwave))
        all_v2_lam.extend(np.tile(eff_wave, nobs))
        all_v2_dlam.extend(np.tile(eff_band, nobs))
        
        # Station indices (2 x nobs*nwave)
        sta_repeated = np.repeat(sta_index, nwave, axis=1)
        all_v2_sta_index.append(sta_repeated)
    
    if not all_v2:
        return None
    
    return {
        'v2': np.concatenate(all_v2),
        'v2_err': np.concatenate(all_v2_err),
        'v2_ucoord': np.array(all_v2_ucoord),
        'v2_vcoord': np.array(all_v2_vcoord),
        'v2_mjd': np.array(all_v2_mjd),
        'v2_lam': np.array(all_v2_lam),
        'v2_dlam': np.array(all_v2_dlam),
        'v2_flag': np.concatenate(all_v2_flag),
        'v2_sta_index': np.hstack(all_v2_sta_index) if all_v2_sta_index else np.zeros((2, 0), dtype=int)
    }


def _parse_oi_t3(hdul, wav_dict, target_filter=None):
    """Parse OI_T3 tables (closure phases and triple amplitudes)."""
    t3_tables = [hdu for hdu in hdul if hdu.name == 'OI_T3']
    if not t3_tables:
        return None
    
    all_t3amp = []
    all_t3amp_err = []
    all_t3phi = []
    all_t3phi_err = []
    all_t3_u1coord = []
    all_t3_v1coord = []
    all_t3_u2coord = []
    all_t3_v2coord = []
    all_t3_mjd = []
    all_t3_lam = []
    all_t3_dlam = []
    all_t3_flag = []
    all_t3_sta_index = []
    
    for table in t3_tables:
        data = table.data
        insname = table.header.get('INSNAME', '')
        
        # Filter by target
        if target_filter is not None:
            mask = np.isin(data['TARGET_ID'], target_filter)
        else:
            mask = np.ones(len(data), dtype=bool)
        
        # Get wavelength info
        wav_info = wav_dict.get(insname, {})
        eff_wave = wav_info.get('eff_wave', np.array([1.0]))
        eff_band = wav_info.get('eff_band', np.array([0.0]))
        
        # Extract data - apply mask first to select rows, then stack
        t3amp = np.array([row for row in data['T3AMP'][mask]])  # shape: (nobs, nwave)
        t3amperr = np.array([row for row in data['T3AMPERR'][mask]])
        t3phi = np.array([row for row in data['T3PHI'][mask]])
        t3phierr = np.array([row for row in data['T3PHIERR'][mask]])
        u1coord = data['U1COORD'][mask]
        v1coord = data['V1COORD'][mask]
        u2coord = data['U2COORD'][mask]
        v2coord = data['V2COORD'][mask]
        mjd = data['MJD'][mask]
        flag = np.array([row for row in data['FLAG'][mask]])  # shape: (nobs, nwave)
        sta_index = np.array([row for row in data['STA_INDEX'][mask]]).T  # shape: (3, nobs)
        
        # Flatten
        nobs, nwave = t3amp.shape
        all_t3amp.append(t3amp.ravel())
        all_t3amp_err.append(t3amperr.ravel())
        all_t3phi.append(t3phi.ravel())
        all_t3phi_err.append(t3phierr.ravel())
        all_t3_flag.append(flag.ravel())
        
        # Repeat metadata
        all_t3_u1coord.extend(np.repeat(u1coord, nwave))
        all_t3_v1coord.extend(np.repeat(v1coord, nwave))
        all_t3_u2coord.extend(np.repeat(u2coord, nwave))
        all_t3_v2coord.extend(np.repeat(v2coord, nwave))
        all_t3_mjd.extend(np.repeat(mjd, nwave))
        all_t3_lam.extend(np.tile(eff_wave, nobs))
        all_t3_dlam.extend(np.tile(eff_band, nobs))
        
        # Station indices
        sta_repeated = np.repeat(sta_index, nwave, axis=1)
        all_t3_sta_index.append(sta_repeated)
    
    if not all_t3amp:
        return None
    
    return {
        't3amp': np.concatenate(all_t3amp),
        't3amp_err': np.concatenate(all_t3amp_err),
        't3phi': np.concatenate(all_t3phi),
        't3phi_err': np.concatenate(all_t3phi_err),
        't3_u1coord': np.array(all_t3_u1coord),
        't3_v1coord': np.array(all_t3_v1coord),
        't3_u2coord': np.array(all_t3_u2coord),
        't3_v2coord': np.array(all_t3_v2coord),
        't3_mjd': np.array(all_t3_mjd),
        't3_lam': np.array(all_t3_lam),
        't3_dlam': np.array(all_t3_dlam),
        't3_flag': np.concatenate(all_t3_flag),
        't3_sta_index': np.hstack(all_t3_sta_index) if all_t3_sta_index else np.zeros((3, 0), dtype=int)
    }


def readoifits(filename, 
               targetname="",
               use_vis=True,
               use_v2=True, 
               use_t3=True,
               filter_bad_data=True,
               redundance_remove=True,
               uvtol=2e2,
               **filter_kwargs):
    """
    Read OIFITS file and return OIData object.
    
    Args:
        filename: Path to OIFITS file
        targetname: Target name to filter (empty string = all targets)
        use_vis: Include VIS (complex visibility) data
        use_v2: Include V2 data
        use_t3: Include T3 data  
        filter_bad_data: Filter out bad data points
        redundance_remove: Remove redundant UV points
        uvtol: UV tolerance for redundancy removal
        **filter_kwargs: Additional arguments for set_data_filter()
        
    Returns:
        OIData object
    """
    with fits.open(filename) as hdul:
        # Extract wavelength info
        wav_dict = _extract_wavelength_info(hdul)
        
        # Get target filter
        target_filter = None
        if targetname:
            target_tables = [hdu for hdu in hdul if hdu.name == 'OI_TARGET']
            if target_tables:
                targ_data = target_tables[0].data
                target_filter = targ_data['TARGET_ID'][targ_data['TARGET'] == targetname]
        
        # Parse VIS data (complex visibilities)
        vis_data = _parse_oi_vis(hdul, wav_dict, target_filter) if use_vis else None
        
        # Parse V2 data
        v2_data = _parse_oi_vis2(hdul, wav_dict, target_filter) if use_v2 else None
        
        # Parse T3 data
        t3_data = _parse_oi_t3(hdul, wav_dict, target_filter) if use_t3 else None
        
        # Build UV coverage and create OIData
        data = _build_oidata(filename, vis_data, v2_data, t3_data)
        
        # Apply filtering
        if filter_bad_data:
            bad_indices = set_data_filter(data, filter_bad_data=True, **filter_kwargs)
            data = filter_data(data, bad_indices)
        
        # Remove redundant UV points
        if redundance_remove:
            data = remove_redundant_uv(data, uvtol=uvtol)
        
        return data


def _build_oidata(filename, vis_data, v2_data, t3_data):
    """Build OIData object from parsed VIS, V2 and T3 data."""
    data = OIData()
    data.filename = filename
    
    # Initialize lists for UV coverage
    uv_list = []
    uv_lam_list = []
    uv_dlam_list = []
    uv_mjd_list = []
    
    offset = 0  # Track UV point offset for indexing
    
    # Process VIS data (complex visibilities) - FIRST, matching Julia order
    if vis_data is not None:
        nvis = len(vis_data['visamp'])
        data.visamp = vis_data['visamp']
        data.visamp_err = vis_data['visamp_err']
        data.visphi = vis_data['visphi']
        data.visphi_err = vis_data['visphi_err']
        data.vis_mjd = vis_data['vis_mjd']
        data.vis_lam = vis_data['vis_lam']
        data.vis_dlam = vis_data['vis_dlam']
        data.vis_flag = vis_data['vis_flag']
        data.vis_sta_index = vis_data['vis_sta_index']
        data.nvisamp = nvis
        data.nvisphi = nvis
        
        # Compute UV coordinates (u, v in wavelengths)
        vis_u = vis_data['vis_ucoord'] / vis_data['vis_lam']
        vis_v = vis_data['vis_vcoord'] / vis_data['vis_lam']
        data.vis_baseline = np.sqrt(vis_u**2 + vis_v**2)
        
        # Add to UV coverage
        uv_list.append(np.vstack([vis_u, vis_v]))
        uv_lam_list.append(vis_data['vis_lam'])
        uv_dlam_list.append(vis_data['vis_dlam'])
        uv_mjd_list.append(vis_data['vis_mjd'])
        
        # Index mapping
        data.indx_vis = np.arange(offset, offset + nvis)
        offset += nvis
    
    # Process V2 data
    if v2_data is not None:
        nv2 = len(v2_data['v2'])
        data.v2 = v2_data['v2']
        data.v2_err = v2_data['v2_err']
        data.v2_mjd = v2_data['v2_mjd']
        data.v2_lam = v2_data['v2_lam']
        data.v2_dlam = v2_data['v2_dlam']
        data.v2_flag = v2_data['v2_flag']
        data.v2_sta_index = v2_data['v2_sta_index']
        data.nv2 = nv2
        
        # Compute UV coordinates (u, v in wavelengths)
        v2_u = v2_data['v2_ucoord'] / v2_data['v2_lam']
        v2_v = v2_data['v2_vcoord'] / v2_data['v2_lam']
        data.v2_baseline = np.sqrt(v2_u**2 + v2_v**2)
        
        # Add to UV coverage
        uv_list.append(np.vstack([v2_u, v2_v]))
        uv_lam_list.append(v2_data['v2_lam'])
        uv_dlam_list.append(v2_data['v2_dlam'])
        uv_mjd_list.append(v2_data['v2_mjd'])
        
        # Index mapping
        data.indx_v2 = np.arange(offset, offset + nv2)
        offset += nv2
    
    # Process T3 data
    if t3_data is not None:
        nt3 = len(t3_data['t3amp'])
        data.t3amp = t3_data['t3amp']
        data.t3amp_err = t3_data['t3amp_err']
        data.t3phi = t3_data['t3phi']
        data.t3phi_err = t3_data['t3phi_err']
        data.t3_mjd = t3_data['t3_mjd']
        data.t3_lam = t3_data['t3_lam']
        data.t3_dlam = t3_data['t3_dlam']
        data.t3_flag = t3_data['t3_flag']
        data.t3_sta_index = t3_data['t3_sta_index']
        data.nt3amp = nt3
        data.nt3phi = nt3
        
        # Compute UV coordinates for all 3 baselines
        t3_u1 = t3_data['t3_u1coord'] / t3_data['t3_lam']
        t3_v1 = t3_data['t3_v1coord'] / t3_data['t3_lam']
        t3_u2 = t3_data['t3_u2coord'] / t3_data['t3_lam']
        t3_v2 = t3_data['t3_v2coord'] / t3_data['t3_lam']
        t3_u3 = t3_u1 + t3_u2  # Third baseline is sum (closure)
        t3_v3 = t3_v1 + t3_v2
        
        b1 = np.sqrt(t3_u1**2 + t3_v1**2)
        b2 = np.sqrt(t3_u2**2 + t3_v2**2)
        b3 = np.sqrt(t3_u3**2 + t3_v3**2)
        data.t3_baseline = (b1 * b2 * b3) ** (1.0/3.0)  # Geometric mean
        data.t3_maxbaseline = np.maximum(np.maximum(b1, b2), b3)
        
        # Add to UV coverage (3 baselines per T3 point)
        uv_list.extend([
            np.vstack([t3_u1, t3_v1]),
            np.vstack([t3_u2, t3_v2]),
            np.vstack([t3_u3, t3_v3])
        ])
        uv_lam_list.extend([t3_data['t3_lam']] * 3)
        uv_dlam_list.extend([t3_data['t3_dlam']] * 3)
        uv_mjd_list.extend([t3_data['t3_mjd']] * 3)
        
        # Index mapping (3 UV points per T3 measurement)
        data.indx_t3_1 = np.arange(offset, offset + nt3)
        data.indx_t3_2 = np.arange(offset + nt3, offset + 2*nt3)
        data.indx_t3_3 = np.arange(offset + 2*nt3, offset + 3*nt3)
        offset += 3 * nt3
    
    # Combine UV coverage
    if uv_list:
        data.uv = np.hstack(uv_list)
        data.uv_lam = np.concatenate(uv_lam_list)
        data.uv_dlam = np.concatenate(uv_dlam_list)
        data.uv_mjd = np.concatenate(uv_mjd_list)
        data.uv_baseline = np.sqrt(data.uv[0]**2 + data.uv[1]**2)
        data.nuv = data.uv.shape[1]
        
        data.mean_mjd = np.mean(data.uv_mjd)
    
    return data


def readoifits_multiepochs(oifitsfiles, filter_bad_data=True, force_full_t3=False, **kwargs):
    """
    Read multiple OIFITS files, each containing a single epoch.
    
    Args:
        oifitsfiles: List of OIFITS file paths
        filter_bad_data: Apply bad data filtering
        force_full_t3: Require both T3amp and T3phi
        **kwargs: Additional arguments for readoifits()
        
    Returns:
        nepochs: Number of epochs
        tepochs: Array of MJD times
        data: List of OIData objects
    """
    nepochs = len(oifitsfiles)
    tepochs = np.zeros(nepochs)
    data = []
    
    for i, filename in enumerate(oifitsfiles):
        oi_data = readoifits(filename, 
                            filter_bad_data=filter_bad_data,
                            force_full_t3=force_full_t3,
                            **kwargs)
        data.append(oi_data)
        tepochs[i] = oi_data.mean_mjd
        print(f"{filename}\t MJD: {tepochs[i]:.3f}\t nV2 = {oi_data.nv2}\t "
              f"nT3amp = {oi_data.nt3amp}\t nT3phi = {oi_data.nt3phi}")
    
    return nepochs, tepochs, data


def readoifits_multicolors(oifitsfiles, filter_bad_data=False, force_full_t3=False, **kwargs):
    """
    Read multiple OIFITS files, each containing a single wavelength.
    
    Args:
        oifitsfiles: List of OIFITS file paths
        filter_bad_data: Apply bad data filtering
        force_full_t3: Require both T3amp and T3phi
        **kwargs: Additional arguments for readoifits()
        
    Returns:
        data: List of OIData objects
    """
    data = []
    
    for filename in oifitsfiles:
        oi_data = readoifits(filename,
                            filter_bad_data=filter_bad_data,
                            force_full_t3=force_full_t3,
                            **kwargs)
        data.append(oi_data)
        print(f"{filename}\t nV2 = {oi_data.nv2}\t nT3amp = {oi_data.nt3amp}\t nT3phi = {oi_data.nt3phi}")
    
    return data


def oifits_prep(data: OIData,
                min_v2_err_add=0.0,
                min_v2_err_rel=0.0,
                v2_err_mult=1.0,
                min_t3amp_err_add=0.0,
                min_t3amp_err_rel=0.0,
                t3amp_err_mult=1.0,
                min_t3phi_err_add=0.0,
                t3phi_err_mult=1.0,
                quad=False):
    """
    Adjust error bars on interferometric data.
    
    Examples (MIRC from Monnier et al. 2012):
        min_v2_err_add=2e-4, min_v2_err_rel=0.066,
        min_t3amp_err_add=1e-5, min_t3amp_err_rel=0.1,
        min_t3phi_err_add=1.0
    
    Args:
        data: OIData object (modified in place)
        min_v2_err_add: Minimum additive V2 error floor
        min_v2_err_rel: Minimum relative V2 error (fraction)
        v2_err_mult: Multiplier for existing V2 errors
        min_t3amp_err_add: Minimum additive T3amp error floor
        min_t3amp_err_rel: Minimum relative T3amp error (fraction)
        t3amp_err_mult: Multiplier for existing T3amp errors
        min_t3phi_err_add: Minimum additive T3phi error floor (degrees)
        t3phi_err_mult: Multiplier for existing T3phi errors
        quad: Use quadrature sum instead of max
        
    Returns:
        data: Modified OIData object
    """
    # Adjust V2 errors
    if data.nv2 > 0:
        if not quad:
            temperr = v2_err_mult * data.v2_err
            newerr = np.abs(data.v2 * min_v2_err_rel) + min_v2_err_add
            data.v2_err = np.maximum(temperr, newerr)
        else:
            data.v2_err = np.sqrt(
                (data.v2 * min_v2_err_rel)**2 +
                (v2_err_mult * data.v2_err)**2 +
                min_v2_err_add**2
            )
    
    # Adjust T3amp errors
    if data.nt3amp > 0:
        if not quad:
            temperr = t3amp_err_mult * data.t3amp_err
            newerr = np.abs(data.t3amp * min_t3amp_err_rel) + min_t3amp_err_add
            data.t3amp_err = np.maximum(temperr, newerr)
        else:
            data.t3amp_err = np.sqrt(
                (data.t3amp * min_t3amp_err_rel)**2 +
                (t3amp_err_mult * data.t3amp_err)**2 +
                min_t3amp_err_add**2
            )
    
    # Adjust T3phi errors (in degrees)
    if data.nt3phi > 0:
        if not quad:
            temperr = t3phi_err_mult * data.t3phi_err
            newerr = np.full(len(data.t3phi_err), min_t3phi_err_add)
            data.t3phi_err = np.maximum(temperr, newerr)
        else:
            data.t3phi_err = np.sqrt(
                (t3phi_err_mult * data.t3phi_err)**2 +
                min_t3phi_err_add**2
            )
    
    return data


def list_oifits_targets(filename):
    """
    List all target names in an OIFITS file.
    
    Args:
        filename: Path to OIFITS file
        
    Returns:
        List of unique target names
    """
    with fits.open(filename) as hdul:
        target_tables = [hdu for hdu in hdul if hdu.name == 'OI_TARGET']
        if not target_tables:
            return []
        
        all_targets = []
        for table in target_tables:
            all_targets.extend(table.data['TARGET'])
        
        return list(set(all_targets))