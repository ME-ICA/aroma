"""Plotting functions for ICA-AROMA."""
import os

import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns


def classification_plot(in_file, out_dir):
    """Generate a figure to show classifications.

    Parameters
    ----------
    in_file : str
        Path to tab-delimited file with feature scores and classification.
    out_dir : str
        Output directory.
    """
    assert isinstance(in_file, str)
    df = pd.read_table(in_file)
    motion_components_df = df.loc[df["classification"] == "rejected"]

    # get counts
    n_components = df.shape[0]
    n_motion_components = motion_components_df.shape[0]
    print(
        "Found {} head motion-related components in a total of {} components.".format(
            n_motion_components, n_components
        )
    )

    # add dummy components if needed, this is just for making the plots look nice
    if n_motion_components < 3:
        temp_df = pd.DataFrame(
            {
                "classification": ["rejected", "rejected", "rejected"],
                "max_RP_corr": [1.0, 1.0, 1.0],
                "edge_fract": [1.0, 1.0, 1.0],
                "HFC": [0.0, 0.0, 0.0],
                "csf_fract": [0.0, 0.0, 0.0],
            }
        )
        df = df.append(temp_df, ignore_index=True)

    if df.loc[df["classification"] == "accepted"].shape[0] < 3:
        temp_df = pd.DataFrame(
            {
                "classification": ["accepted", "accepted", "accepted"],
                "max_RP_corr": [0.0, 0.0, 0.0],
                "edge_fract": [0.0, 0.0, 0.0],
                "HFC": [0.0, 0.0, 0.0],
                "csf_fract": [0.0, 0.0, 0.0],
            }
        )
        df = df.append(temp_df, ignore_index=True)

    # rename columns
    df = df.rename(
        columns={
            "classification": "Motion",
            "max_RP_corr": "RP",
            "edge_fract": "Edge",
            "HFC": "Freq",
            "csf_fract": "CSF",
        }
    )
    df["classification"] = df["classification"].map(
        {"rejected": "True", "accepted": "False"}
    )

    # Make pretty figure
    # styling
    sns.set_style("white")
    colortrue = "#FFBF17"
    colorfalse = "#69A00A"

    # create figure
    fig = plt.figure(figsize=[12, 4])

    # define grids
    gs = gridspec.GridSpec(4, 7, wspace=1)
    gs00 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[:, 0:3])
    gs01 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 3:5])
    gs02 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 5:7])

    # define subplots
    # Edge/RP
    ax1 = fig.add_subplot(gs00[1:4, 0:3])
    # distribution edge (ax1 top)
    ax1t = fig.add_subplot(gs00[0, 0:3])
    # distribution RP (ax1 right)
    ax1r = fig.add_subplot(gs00[1:4, 3])
    # Freq
    ax2 = fig.add_subplot(gs01[1:4, :])
    # CSF
    ax3 = fig.add_subplot(gs02[1:4, :])

    # plot Freq
    sns.boxplot(
        x="Motion",
        y="Freq",
        data=df,
        ax=ax2,
        palette=[colortrue, colorfalse],
        order=["True", "False"],
    )
    ax2.hlines(0.35, -1, 2, zorder=0, linestyles="dotted", linewidth=0.5)
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Classification", fontsize=14, labelpad=10)
    ax2.set_ylabel("High-Frequency Content", fontsize=14)
    ax2.set_xticklabels(["Motion", "Other"])
    ax2.tick_params(axis="both", labelsize=12)
    sns.despine(ax=ax2)

    # plot CSF
    sns.boxplot(
        x="Motion",
        y="CSF",
        data=df,
        ax=ax3,
        palette=[colortrue, colorfalse],
        order=["True", "False"],
    )
    ax3.hlines(0.1, -1, 2, zorder=0, linestyles="dotted", linewidth=0.5)
    ax3.set_ylim([0, 1])
    ax3.set_xlabel("Classification", fontsize=14, labelpad=10)
    ax3.set_ylabel("CSF Fraction", fontsize=14)
    ax3.set_xticklabels(["Motion", "Other"])
    ax3.tick_params(axis="both", labelsize=12)
    sns.despine(ax=ax3)

    # plot Edge/RP relationship
    # obtain projection line
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])
    a = -hyp[1] / hyp[2]
    xx = np.linspace(0, 1)
    yy = a * xx - hyp[0] / hyp[2]
    # plot scatter and line
    if len(df) > 100:
        sizemarker = 6
    else:
        sizemarker = 10
    ax1.scatter(
        x="RP",
        y="Edge",
        data=df.loc[df["Motion"] == "False"],
        color=colorfalse,
        s=sizemarker,
    )
    # plot true ones on top to see how much the go over the border
    # this gives an indication for how many were selected using the
    # two other features
    ax1.scatter(
        x="RP",
        y="Edge",
        data=df.loc[df["Motion"] == "True"],
        color=colortrue,
        s=sizemarker,
    )
    # add decision boundary
    ax1.plot(xx, yy, ".", color="k", markersize=1)
    # styling
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])
    ax1.set_xlabel("Maximum RP Correlation", fontsize=14, labelpad=10)
    ax1.set_ylabel("Edge Fraction", fontsize=14)
    ax1.set_xticks(np.arange(0, 1.2, 0.2))
    ax1.set_yticks(np.arange(0, 1.2, 0.2))
    ax1.tick_params(axis="both", labelsize=12)

    # plot distributions
    # RP
    sns.distplot(
        df.loc[df["Motion"] == "True", "RP"],
        ax=ax1t,
        color=colortrue,
        hist_kws={"alpha": 0.2},
    )
    sns.distplot(
        df.loc[df["Motion"] == "False", "RP"],
        ax=ax1t,
        color=colorfalse,
        hist_kws={"alpha": 0.2},
    )
    ax1t.set_xlim([0, 1])

    # Edge
    sns.distplot(
        df.loc[df["Motion"] == "True", "Edge"],
        ax=ax1r,
        vertical=True,
        color=colortrue,
        hist_kws={"alpha": 0.2},
    )
    sns.distplot(
        df.loc[df["Motion"] == "False", "Edge"],
        ax=ax1r,
        vertical=True,
        color=colorfalse,
        hist_kws={"alpha": 0.2},
    )
    ax1r.set_ylim([0, 1])

    # cosmetics
    for myax in [ax1t, ax1r]:
        myax.set_xticks([])
        myax.set_yticks([])
        myax.set_xlabel("")
        myax.set_ylabel("")
        myax.spines["right"].set_visible(False)
        myax.spines["top"].set_visible(False)
        myax.spines["bottom"].set_visible(False)
        myax.spines["left"].set_visible(False)

    # bring tickmarks back
    for myax in fig.get_axes():
        myax.tick_params(which="major", direction="in", length=3)

    # add figure title
    plt.suptitle("Component Assessment", fontsize=20)

    # outtakes
    plt.savefig(
        os.path.join(out_dir, "ICA_AROMA_component_assessment.svg"),
        bbox_inches="tight",
        dpi=300,
    )
