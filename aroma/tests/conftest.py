import os.path as op
import pytest

from nilearn.datasets import fetch_development_fmri


import aroma

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


@pytest.fixture(scope="session")
def testpath(tmp_path_factory):
    """ Test path that will be used to download all files """
    return tmp_path_factory.getbasetemp()


@pytest.fixture
def featurespath():
    return op.join(op.dirname(aroma.__file__), 'tests', 'data', 'features_test')


@pytest.fixture
def nilearn_data(testpath):
    return fetch_development_fmri(n_subjects=1, age_group="adult")


@pytest.fixture
def mel_FT_mix(featurespath):
    return op.join(featurespath, 'melodic_FTmix')


@pytest.fixture
def mel_mix(featurespath):
    return op.join(featurespath, 'melodic_mix')


@pytest.fixture
def mc(featurespath):
    return op.join(featurespath, 'mc.tsv')


@pytest.fixture
def mel_IC(featurespath):
    return op.join(featurespath, 'melodic_IC_thr_MNI2mm.nii.gz')


@pytest.fixture
def features_df(featurespath):
    return op.join(featurespath, 'features.csv')
