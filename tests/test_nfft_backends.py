import numpy as np

from moifits.oichi2 import NFFTPlan, image_to_vis, setup_nfft


class DummyData:
    uv = np.array([[0.0, 1.0], [0.0, 0.0]])
    indx_vis = np.array([0, 1])
    indx_v2 = np.array([0, 1])
    indx_t3_1 = np.array([0])
    indx_t3_2 = np.array([0])
    indx_t3_3 = np.array([0])


def test_direct_backend_forward_zero_baseline_is_normalized_flux():
    plan = NFFTPlan(np.array([[0.0], [0.0]]), nx=4, pixsize_mas=0.125, backend="direct")
    image = np.ones((4, 4))

    vis = image_to_vis(image, plan)

    np.testing.assert_allclose(vis, [1.0 + 0.0j], atol=1e-12)


def test_setup_nfft_propagates_backend():
    plans = setup_nfft(DummyData(), nx=4, pixsize=0.125, backend="direct")

    assert all(plan.backend == "direct" for plan in plans)


if __name__ == "__main__":
    test_direct_backend_forward_zero_baseline_is_normalized_flux()
    test_setup_nfft_propagates_backend()
