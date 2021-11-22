"""Integration tests for aroma."""
import os.path as op

import numpy as np
import pandas as pd
import pytest
from aroma.aroma import aroma_workflow


def test_integration(
    skip_integration, nilearn_data, mel_mix, mel_IC, classification_overview, classified_motion_ICs
):
    """Perform integration test."""
    if skip_integration:
        pytest.skip("Skipping integration test")

    # Obtain test path
    test_path, _ = op.split(nilearn_data.func[0])

    # Create output path
    out_path = op.join(test_path, "out")

    # Read confounds
    confounds = pd.read_table(nilearn_data.confounds[0])

    # Extract motion parameters from confounds
    mc = confounds[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
    mc_path = op.join(test_path, "mc.txt")
    mc.to_csv(mc_path, sep="\t", index=False, header=None)

    # Add seed for reproducibility
    np.random.seed(42)

    aroma_workflow(
        in_file=nilearn_data.func[0],
        mixing=mel_mix,
        component_maps=mel_IC,
        mc=mc_path,
        out_dir=out_path,
        TR=2,
        den_type="nonaggr",
        generate_plots=False,
        overwrite=True,
    )

    # Make sure files are generated
    assert op.isfile(op.join(out_path, "desc-AROMA_metrics.tsv"))
    assert op.isfile(op.join(out_path, "AROMAnoiseICs.csv"))
    assert op.isfile(op.join(out_path, "desc-smoothAROMAnonaggr_bold.nii.gz"))

    # Load classification overview file
    true_classification_overview = pd.read_table(
        classification_overview,
        index_col="IC",
    )
    test_classification_overview = pd.read_table(
        op.join(out_path, "desc-AROMA_metrics.tsv"),
        index_col="IC",
    )

    #  Check feature scores
    f_scores = test_classification_overview[["edge_fract", "csf_fract", "max_RP_corr", "HFC"]]
    f_true = true_classification_overview[["edge_fract", "csf_fract", "max_RP_corr", "HFC"]]
    assert np.allclose(f_true.values, f_scores.values, atol=0.01), f_true.values - f_scores.values

    # Check classifications
    assert (
        true_classification_overview["classification"].tolist()
        == test_classification_overview["classification"].tolist()
    )

    # Check motion ICs
    test_mot_ics = np.loadtxt(op.join(out_path, "AROMAnoiseICs.csv"), delimiter=",")
    true_mot_ics = np.loadtxt(classified_motion_ICs, delimiter=",")
    assert np.allclose(true_mot_ics, test_mot_ics)
