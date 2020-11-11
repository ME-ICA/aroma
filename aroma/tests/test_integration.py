import numpy as np
import pandas as pd
from os.path import join, split, isfile

from aroma.aroma import aroma_workflow
from aroma.tests.utils import get_tests_resource_path

import pytest


def test_integration(skip_integration, nilearn_data):
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
    assert isfile(join(out_path, "classification_overview.txt"))
    assert isfile(join(out_path, "classified_motion_ICs.txt"))
    assert isfile(join(out_path, "denoised_func_data_nonaggr.nii.gz"))
    assert isfile(join(out_path, "feature_scores.txt"))
    assert isfile(join(out_path, "mask.nii.gz"))
    assert isfile(join(out_path, "melodic_IC_thr.nii.gz"))
    assert isfile(join(out_path, "melodic_IC_thr_MNI2mm.nii.gz"))

    # Check classification overview file
    true_classification_overview = pd.read_csv(
        join(resources_path, "classification_overview.txt"),
        sep="\t",
        index_col="IC",
        nrows=4,
    )
    classification_overview = pd.read_csv(
        join(out_path, "classification_overview.txt"), sep="\t", index_col="IC", nrows=4
    )

    assert np.allclose(
        true_classification_overview.iloc[:, 1:],
        classification_overview.iloc[:, 1:],
        atol=0.05,
    )

    #  Check feature scores
    f_scores = np.loadtxt(join(out_path, "feature_scores.txt"))
    f_true = np.loadtxt(join(resources_path, "feature_scores.txt"))
    assert np.allclose(f_true[0, :], f_scores[0, :], atol=0.01)

    # Check motion ICs
    mot_ics = np.loadtxt(join(out_path, "classified_motion_ICs.txt"), delimiter=",")
    true_mot_ics = np.loadtxt(
        join(resources_path, "classified_motion_ICs.txt"), delimiter=","
    )
    assert np.allclose(true_mot_ics[:4], mot_ics[:4])
