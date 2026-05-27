import numpy as np
from scipy.special import j0, j1

def visibility_ud(diameter, uv):
    """
    Compute visibility of a uniform disk.
    
    Args:
        diameter: List containing [diameter_in_mas]
        uv: UV coordinates array (2, N)
        
    Returns:
        Complex visibility array
    """
    # Convert diameter from mas to radians
    diam_rad = diameter[0] * (np.pi / 180.0) / 3600000.0
    
    # Compute baseline lengths
    baseline = np.sqrt(uv[0]**2 + uv[1]**2)
    
    # Uniform disk visibility: 2*J1(pi*B*d)/(pi*B*d)
    x = np.pi * baseline * diam_rad
    
    # Handle x=0 case
    vis = np.ones_like(x)
    mask = x != 0
    vis[mask] = 2.0 * j1(x[mask]) / x[mask]
    
    return vis

def dvisibility_ud(diameter, uv):
    """
    Compute derivative of uniform disk visibility w.r.t. diameter.
    
    Args:
        diameter: List containing [diameter_in_mas]
        uv: UV coordinates array (2, N)
        
    Returns:
        Derivative of visibility w.r.t. diameter
    """
    diam_rad = diameter[0] * (np.pi / 180.0) / 3.6e6
    baseline = np.sqrt(uv[0]**2 + uv[1]**2)
    x = np.pi * baseline * diam_rad
    dvis = np.zeros_like(x)
    mask = x != 0

    j0_vals = j0(x[mask])
    j1_vals = j1(x[mask])
    x_vals = x[mask]

    # Match Julia: dV/dt = 2 * (t*j0 - 2*j1)/t^2 ; dV/dD = dV/dt * dt/dD
    dvis[mask] = 2.0 * (x_vals * j0_vals - 2.0 * j1_vals) / (x_vals**2) \
                 * np.pi * baseline[mask] * (np.pi / 180.0) / 3.6e6

    return dvis