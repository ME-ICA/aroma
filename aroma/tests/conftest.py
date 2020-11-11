import os
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


@pytest.fixture(scope="session")
def testpath(tmp_path_factory):
    """ Test path that will be used to download all files """
    return tmp_path_factory.getbasetemp()


@pytest.fixture
def nilearn_data(testpath):
    return fetch_development_fmri(n_subjects=1, age_group="adult")
