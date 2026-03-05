import numpy as np
import sys
from pathlib import Path

# Import local modules directly from moifits/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from writeoifits import generate_uv_sampling
from image_to_observables import image_to_cvis_grid, sample_image_observables, create_oifits_from_image

# 1) Make UV sampling
stations = np.array([[0,0,0],[40,0,0],[0,60,0],[55,55,0]], dtype=float)
ha = np.linspace(-1.0, 1.0, 10)  # rad
mjd = 60000.0 + np.arange(ha.size) * 600.0 / 86400.0
sampling = generate_uv_sampling(stations, ha, dec_rad=np.deg2rad(30.0), mjd_times=mjd)

# 2) Image -> cvis
image = np.ones((128, 128), dtype=float)  # replace with your model image
waves = np.array([1.60e-6, 1.65e-6, 1.70e-6])
cvis = image_to_cvis_grid(image, pixsize_mas=0.1,
                          ucoord_m=sampling["vis_ucoord"], vcoord_m=sampling["vis_vcoord"],
                          wavelengths_m=waves)

# 3) Image -> observables
obs = sample_image_observables(image, pixsize_mas=0.1, sampling=sampling, wavelengths_m=waves)

# 4) Image -> OIFITS
create_oifits_from_image("synthetic_from_image.oifits", image=image, pixsize_mas=0.1,
                         station_enu_m=stations, hour_angles_rad=ha, dec_rad=np.deg2rad(30.0),
                         wavelengths_m=waves)
