"""Input/output functions for the Aroma project."""
import json
import os.path as op


def write_metrics(features_df, out_dir, metric_metadata=None):
    """Write out feature/classification information and metadata.
     Parameters
    ----------
    features_df : (C x 5) :obj:`pandas.DataFrame`
        DataFrame with metric values and classifications.
        Must have the following columns: "edge_fract", "csf_fract", "max_RP_corr", "HFC", and
        "classification".
    out_dir : :obj:`str`
        Output directory.
    metric_metadata : :obj:`dict` or None, optional
        Metric metadata in a dictionary.
     Returns
    -------
    motion_ICs : array_like
        Array containing the indices of the components identified as motion components.
     Output
    ------
    AROMAnoiseICs.csv : A text file containing the indices of the
                        components identified as motion components
    desc-AROMA_metrics.tsv
    desc-AROMA_metrics.json
    """
    # Put the indices of motion-classified ICs in a text file (starting with 1)
    motion_ICs = features_df["classification"][features_df["classification"] == "rejected"].index
    motion_ICs = motion_ICs.values

    with open(op.join(out_dir, "AROMAnoiseICs.csv"), "w") as fo:
        out_str = ",".join(motion_ICs.astype(str))
        fo.write(out_str)

    # Create a summary overview of the classification
    out_file = op.join(out_dir, "desc-AROMA_metrics.tsv")
    features_df.to_csv(out_file, sep="\t", index_label="IC")

    if isinstance(metric_metadata, dict):
        with open(op.join(out_dir, "desc-AROMA_metrics.json"), "w") as fo:
            json.dump(metric_metadata, fo, sort_keys=True, indent=4)

    return motion_ICs
