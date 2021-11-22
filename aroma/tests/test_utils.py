"""Tests for aroma.utils."""
import os

import numpy as np
import pandas as pd
import pytest
from aroma import utils


def test_classification(classification_overview):
    """Test aroma.utils.classification and ensure classifications come out the same."""
    clf_overview_df = pd.read_table(classification_overview)
    test_df = clf_overview_df[["edge_fract", "csf_fract", "max_RP_corr", "HFC"]]
    test_df, metadata = utils.classification(test_df, {})
    true_classifications = clf_overview_df["classification"].tolist()
    test_classifications = test_df["classification"].tolist()
    assert true_classifications == test_classifications


def test_load_motpars_manual(motion_parameters):
    """Test aroma.utils.load_motpars with manual source determination."""
    fsl = utils.load_motpars(motion_parameters["FSL"], source="fsl")
    afni = utils.load_motpars(motion_parameters["AfNI"], source="afni")
    spm = utils.load_motpars(motion_parameters["SPM"], source="spm")
    fmriprep = utils.load_motpars(motion_parameters["fMRIPrep"], source="fmriprep")
    assert np.allclose(fsl, afni)
    assert np.allclose(fsl, spm)
    assert np.allclose(fsl, fmriprep)


def test_load_motpars_auto(motion_parameters):
    """Test aroma.utils.load_motpars with automatic source determination."""
    fsl = utils.load_motpars(motion_parameters["FSL"], source="auto")
    afni = utils.load_motpars(motion_parameters["AfNI"], source="auto")
    spm = utils.load_motpars(motion_parameters["SPM"], source="auto")
    fmriprep = utils.load_motpars(motion_parameters["fMRIPrep"], source="auto")
    assert np.allclose(fsl, afni)
    assert np.allclose(fsl, spm)
    assert np.allclose(fsl, fmriprep)


def test_load_motpars_break(motion_parameters):
    """Break aroma.utils.load_motpars."""
    with pytest.raises(Exception):
        utils.load_motpars("dog.dog", source="auto")

    with pytest.raises(ValueError):
        utils.load_motpars(motion_parameters["FSL"], source="dog")


def test_motpars_fmriprep2fsl(motion_parameters):
    """Test aroma.utils.motpars_fmriprep2fsl."""
    fsl = utils.load_motpars(motion_parameters["FSL"], source="fsl")
    fmriprep = utils.motpars_fmriprep2fsl(motion_parameters["fMRIPrep"])
    assert np.allclose(fsl, fmriprep)

    with pytest.raises(ValueError):
        utils.motpars_fmriprep2fsl(5)

    bad_data = np.random.random((200, 7))
    with pytest.raises(ValueError):
        utils.motpars_fmriprep2fsl(bad_data)


def test_motpars_spm2fsl(motion_parameters):
    """Test aroma.utils.motpars_spm2fsl."""
    fsl = utils.load_motpars(motion_parameters["FSL"], source="fsl")
    spm = utils.motpars_spm2fsl(motion_parameters["SPM"])
    assert np.allclose(fsl, spm)

    with pytest.raises(ValueError):
        utils.motpars_spm2fsl(5)

    bad_data = np.random.random((200, 7))
    with pytest.raises(ValueError):
        utils.motpars_spm2fsl(bad_data)


def test_motpars_afni2fsl(motion_parameters):
    """Test aroma.utils.motpars_afni2fsl."""
    fsl = utils.load_motpars(motion_parameters["FSL"], source="fsl")
    afni = utils.motpars_afni2fsl(motion_parameters["AfNI"])
    assert np.allclose(fsl, afni)

    with pytest.raises(ValueError):
        utils.motpars_afni2fsl(5)

    bad_data = np.random.random((200, 7))
    with pytest.raises(ValueError):
        utils.motpars_afni2fsl(bad_data)


def test_cross_correlation():
    """Test aroma.utils.cross_correlation."""
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
