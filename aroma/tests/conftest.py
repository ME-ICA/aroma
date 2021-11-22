import os
import ssl
from urllib.request import urlretrieve

import aroma
import pytest
from nilearn.datasets import fetch_development_fmri


def pytest_addoption(parser):
    parser.addoption(
        "--skipintegration",
        action="store_true",
        default=False,
        help="Skip integration tests.",
    )


@pytest.fixture
def skip_integration(request):
    return request.config.getoption("--skipintegration")


def fetch_file(osf_id, path, filename):
    """
    Fetches file located on OSF and downloads to `path`/`filename`.

    Parameters
    ----------
    osf_id : str
        Unique OSF ID for file to be downloaded. Will be inserted into relevant
        location in URL: https://osf.io/{osf_id}/download
    path : str
        Path to which `filename` should be downloaded. Ideally a temporary
        directory
    filename : str
        Name of file to be downloaded (does not necessarily have to match name
        of file on OSF)

    Returns
    -------
    full_path : str
        Full path to downloaded `filename`
    """
    # This restores the same behavior as before.
    # this three lines make tests dowloads work in windows
    if os.name == "nt":
        orig_sslsocket_init = ssl.SSLSocket.__init__
        ssl.SSLSocket.__init__ = (
            lambda *args, cert_reqs=ssl.CERT_NONE, **kwargs: orig_sslsocket_init(
                *args, cert_reqs=ssl.CERT_NONE, **kwargs
            )
        )
        ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://osf.io/{}/download".format(osf_id)
    full_path = os.path.join(path, filename)
    if not os.path.isfile(full_path):
        urlretrieve(url, full_path)
    return full_path


@pytest.fixture(scope="session")
def testpath(tmp_path_factory):
    """Test path that will be used to download all files."""
    return tmp_path_factory.mktemp(basename="data", numbered=False)


@pytest.fixture
def nilearn_data(testpath):
    return fetch_development_fmri(n_subjects=1, age_group="adult", data_dir=str(testpath))


# Feature outputs generated with the following command (adding breakpoints to save results)
# python2 ICA_AROMA.py -o out -i
# sub-pixar123_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
# -mc mc.tsv -tr 2 -np


@pytest.fixture
def mel_FT_mix(testpath):
    return fetch_file("ezhfg", testpath, "melodic_FTmix")


@pytest.fixture
def mel_mix(testpath):
    return fetch_file("69j2h", testpath, "melodic_mix")


@pytest.fixture
def mc(testpath):
    return fetch_file("af275", testpath, "mc_fmriprep.tsv")


@pytest.fixture
def mel_IC(testpath):
    return fetch_file("bw3mq", testpath, "melodic_IC_thr_MNI2mm.nii.gz")


@pytest.fixture
def csfFract(testpath):
    return fetch_file("d298c", testpath, "csfFract.npy")


@pytest.fixture
def edgeFract(testpath):
    return fetch_file("vxdu6", testpath, "edgeFract.npy")


@pytest.fixture
def max_correls(testpath):
    return fetch_file("hpd2m", testpath, "max_correls.npy")


@pytest.fixture
def HFC(testpath):
    return fetch_file("rymwg", testpath, "HFC.npy")


@pytest.fixture
def classification_overview(testpath):
    return fetch_file("2hv3a", testpath, "classification_overview.txt")


@pytest.fixture
def classified_motion_ICs(testpath):
    return fetch_file("jnhgc", testpath, "AROMAnoiseICs.csv")


@pytest.fixture
def feature_scores(testpath):
    return fetch_file("cxwfk", testpath, "feature_scores.txt")


@pytest.fixture
def motion_parameters(testpath):
    """Motion parameter outputs in different formats.

    All outputs manually converted from FSL version.
    """
    files = {
        "FSL": fetch_file("ahtrv", testpath, "mc_fsl.txt"),
        "AfNI": fetch_file("p6ybt", testpath, "mc_afni.1D"),
        "SPM": fetch_file("ct6q4", testpath, "rp_mc_spm.txt"),
        "fMRIPrep": fetch_file("af275", testpath, "mc_fmriprep.tsv"),
    }
    return files
