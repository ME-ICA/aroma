from os.path import isfile, join, split

import numpy as np
import pandas as pd
import pytest
from aroma.aroma import aroma_workflow


def test_integration(skip_integration, nilearn_data, classification_overview,
                     classified_motion_ICs, feature_scores):
    if skip_integration:
        pytest.skip("Skipping integration test")

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
    assert isfile(join(out_path, "classification_overview.txt"))
    assert isfile(join(out_path, "classified_motion_ICs.txt"))
    assert isfile(join(out_path, "denoised_func_data_nonaggr.nii.gz"))
    assert isfile(join(out_path, "feature_scores.tsv"))

    # Check classification overview file
    true_classification_overview = pd.read_csv(classification_overview,
                                               sep="\t",
                                               index_col="IC",
                                               nrows=4,)
    classification_overview = pd.read_csv(
        join(out_path, "classification_overview.txt"), sep="\t", index_col="IC", nrows=4
    )

    assert np.allclose(true_classification_overview.iloc[:, :-1].values,
                       classification_overview.iloc[:, :-1].values,
                       atol=0.9)

    #  Check feature scores
    f_scores = pd.read_table(join(out_path, "feature_scores.tsv"))
    f_true = pd.read_table(feature_scores)
    assert np.allclose(f_true.values, f_scores.values, atol=0.9)

    # Check motion ICs
    mot_ics = np.loadtxt(join(out_path, "classified_motion_ICs.txt"), delimiter=",")
    true_mot_ics = np.loadtxt(classified_motion_ICs, delimiter=",")
    assert np.allclose(true_mot_ics[:4], mot_ics[:4])
