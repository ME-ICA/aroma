import numpy as np

from nilearn import image, masking
from threadpoolctl import threadpool_limits

from aroma import utils


def test_cross_correlation():

    np.random.seed(5)
    a = np.random.rand(4, 4)
    b = np.random.rand(2, 4)

    true_cross_corr = np.array(
        [
            [-0.28624708, -0.62178458],
            [0.37905408, -0.51091252],
            [0.24162976, -0.13454275],
            [0.69255319, 0.07156853],
        ]
    )

    cross_corr = utils.cross_correlation(a.T, b.T)

    assert np.allclose(cross_corr, true_cross_corr)

def test_run_ica(nilearn_data):
    in_file = nilearn_data.func[0]
    mask = masking.compute_epi_mask(in_file)
    smoothed_img = image.smooth_img(in_file, fwhm=8)
    t_r = 2.
    with threadpool_limits(limits=1, user_api=None):
        components, mixing, ft = utils.run_ica(smoothed_img, mask, t_r=t_r)
