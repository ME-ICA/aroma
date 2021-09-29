"""Integration tests for aroma."""
from os.path import isfile, join, split

import numpy as np
import pandas as pd
import pytest
from aroma.aroma import aroma_workflow
from aroma.tests.utils import get_tests_resource_path


def test_integration(skip_integration, nilearn_data):
    """Perform integration test."""
    if skip_integration:
        pytest.skip("Skipping integration test")

    resources_path = get_tests_resource_path()

    # Obtain test path
    test_path, _ = split(nilearn_data.func[0])

    # Create output path
    out_path = join(test_path, "out")

    # Read confounds
    confounds = pd.read_csv(nilearn_data.confounds[0], sep="\t")

    # Extract motion parameters from confounds
    mc = confounds[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
    mc_path = join(test_path, "mc.txt")
    mc.to_csv(mc_path, sep="\t", index=False, header=None)

    mixing = join(resources_path, "melodic_mix")
    component_maps = join(resources_path, "melodic_IC_thr_MNI2mm.nii.gz")

    aroma_workflow(
        in_file=nilearn_data.func[0],
        mixing=mixing,
        component_maps=component_maps,
        mc=mc_path,
        out_dir=out_path,
        TR=2,
        den_type="nonaggr",
        generate_plots=False,
        overwrite=True,
    )

    # Make sure files are generated
    assert isfile(join(out_path, "desc-AROMA_metrics.tsv"))
    assert isfile(join(out_path, "AROMAnoiseICs.csv"))
    assert isfile(join(out_path, "desc-smoothAROMAnonaggr_bold.nii.gz"))

    # Load classification overview file
    true_classification_overview = pd.read_table(
        join(resources_path, "classification_overview.txt"),
        index_col="IC",
    )
    classification_overview = pd.read_table(
        join(out_path, "desc-AROMA_metrics.tsv"),
        index_col="IC",
    )

    #  Check feature scores
    f_scores = classification_overview[["edge_fract", "csf_fract", "max_RP_corr", "HFC"]]
    f_true = true_classification_overview[["edge_fract", "csf_fract", "max_RP_corr", "HFC"]]
    assert np.allclose(f_true.values, f_scores.values), f_true.values - f_scores.values

    # Check classifications
    assert (
        true_classification_overview["classification"].tolist()
        == classification_overview["classification"].tolist()
    )

    # Check motion ICs
    mot_ics = np.loadtxt(join(out_path, "AROMAnoiseICs.csv"), delimiter=",")
    true_mot_ics = np.loadtxt(
        join(resources_path, "AROMAnoiseICs.csv"), delimiter=","
    )
    assert np.allclose(true_mot_ics[:4], mot_ics[:4])
