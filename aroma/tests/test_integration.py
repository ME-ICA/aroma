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

    aroma_workflow(
        TR=2,
        affmat=None,
        den_type="nonaggr",
        dim=0,
        generate_plots=False,
        in_feat=None,
        in_file=nilearn_data.func[0],
        mask=None,
        mc=mc_path,
        mel_dir=None,
        out_dir=out_path,
        overwrite=True,
        warp=None,
    )

    # Make sure files are generated
    assert isfile(join(out_path, "desc-AROMA_metrics.tsv"))
    assert isfile(join(out_path, "classified_motion_ICs.txt"))
    assert isfile(join(out_path, "denoised_func_data_nonaggr.nii.gz"))
    assert isfile(join(out_path, "mask.nii.gz"))
    assert isfile(join(out_path, "melodic_IC_thr.nii.gz"))
    assert isfile(join(out_path, "melodic_IC_thr_MNI2mm.nii.gz"))

    # Check classification overview file
    true_classification_overview = pd.read_table(
        join(resources_path, "classification_overview.txt"),
        index_col="IC",
        nrows=4,
    )
    classification_overview = pd.read_table(
        join(out_path, "desc-AROMA_metrics.tsv"),
        index_col="IC",
        nrows=4,
    )

    assert np.allclose(true_classification_overview.iloc[:, :-1].values,
                       classification_overview.iloc[:, :-1].values,
                       atol=0.9)
    assert np.array_equal(
        true_classification_overview["classification"].values,
        classification_overview["classification"].values,
    )

    #  Check feature scores
    f_scores = classification_overview[["edge_fract", "csf_fract", "max_RP_corr", "HFC"]]
    f_true = pd.read_table(join(resources_path, "feature_scores.txt"))
    assert np.allclose(f_true.values, f_scores.values, atol=0.9)

    # Check motion ICs
    mot_ics = np.loadtxt(join(out_path, "classified_motion_ICs.txt"), delimiter=",")
    true_mot_ics = np.loadtxt(
        join(resources_path, "classified_motion_ICs.txt"), delimiter=","
    )
    assert np.allclose(true_mot_ics[:4], mot_ics[:4])
