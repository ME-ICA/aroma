"""Plotting functions for ICA-AROMA."""
import logging
import os
import glob

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import matplotlib.image as mpimg

mpl.use('Agg')
LGR = logging.getLogger(__name__)


def classification_plot(myinput, out_dir):
    """Generate a figure to show classifications.

    Parameters
    ----------
    myinput
    out_dir
    """
    # find files
    myfiles = glob.glob(myinput)
    LGR.info('Found', len(myfiles), 'file(s)')

    # load in data from files
    count = 0
    for m in myfiles:

        res = []

        tmp = open(m, 'r').read().split('\n')

        for t in tmp[1:-1]:
            vals = t.split('\t')
            res.append([vals[1],
                       float(vals[2]),
                       float(vals[3]),
                       float(vals[4]),
                       float(vals[5])])

        if count == 0:
            df = pd.DataFrame.from_records(res)
        else:
            df2 = pd.DataFrame.from_records(res)
            df = df.append(df2, ignore_index=True)

        count += 1

    # get counts
    ncomp = len(df)
    nmot = len(df.loc[df[0] == "True"])
    LGR.info('Found', nmot, 'head motion-related components in a total of',
             ncomp, 'components.')

    # add dummy components if needed,
    # this is just for making the plots look nice
    tmp = df.loc[df[0] == "True"]
    if len(tmp) < 3:
        df3 = pd.DataFrame.from_records([["True", 1., 1., 0., 0.],
                                        ["True", 1., 1., 0., 0.],
                                        ["True", 1., 1., 0., 0.]])
        df = df.append(df3, ignore_index=True)
    tmp = df.loc[df[0] == "False"]
    if len(tmp) < 3:
        df3 = pd.DataFrame.from_records([["False", 0., 0., 0., 0.],
                                        ["False", 0., 0., 0., 0.],
                                        ["False", 0., 0., 0., 0.]])
        df = df.append(df3, ignore_index=True)

    # rename columns
    df = df.rename(index=str, columns={0: 'Motion',
                                       1: 'RP',
                                       2: 'Edge',
                                       3: 'Freq',
                                       4: 'CSF'})

    # Make pretty figure
    # styling
    sns.set_style('white')
    colortrue = "#FFBF17"
    colorfalse = "#69A00A"
    # plot Edge/RP relationship
    # obtain projection line
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])
    a = -hyp[1] / hyp[2]
    xx = np.linspace(0, 1)
    yy = a * xx - hyp[0] / hyp[2]
    # create aux figure
    h = sns.jointplot(x="RP",
                      y="Edge",
                      data=df,
                      hue=df['Motion'],
                      kind='scatter',
                      palette=[colortrue, colorfalse],
                      hue_order=['True', 'False'],
                      xlim=[0, 1],
                      ylim=[0, 1])
    h.set_axis_labels('Maximum RP Correlation', 'Edge Fraction', fontsize=14, labelpad=10)
    h.ax_joint.set_xticks(np.arange(0, 1.2, 0.2))
    h.ax_joint.set_yticks(np.arange(0, 1.2, 0.2))
    h.ax_joint.tick_params(axis='both', labelsize=12)
    h.ax_joint.plot(xx, yy, '.', color="k", markersize=1)
    h.savefig(os.path.join(out_dir, 'aux_fig.png'),
              bbox_inches='tight', dpi=300)
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
    sns.boxplot(x="Motion",
                y="Freq",
                data=df,
                ax=ax2,
                palette=[colortrue, colorfalse],
                order=['True', 'False'])
    ax2.hlines(0.35, -1, 2, zorder=0, linestyles='dotted', linewidth=0.5)
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Classification', fontsize=14, labelpad=10)
    ax2.set_ylabel('High-Frequency Content', fontsize=14)
    ax2.set_xticklabels(['Motion', 'Other'])
    ax2.tick_params(axis='both', labelsize=12)
    sns.despine(ax=ax2)

    # plot CSF
    sns.boxplot(x="Motion",
                y="CSF",
                data=df,
                ax=ax3,
                palette=[colortrue, colorfalse],
                order=['True', 'False'])
    ax3.hlines(0.1, -1, 2, zorder=0, linestyles='dotted', linewidth=0.5)
    ax3.set_ylim([0, 1])
    ax3.set_xlabel('Classification', fontsize=14, labelpad=10)
    ax3.set_ylabel('CSF Fraction', fontsize=14)
    ax3.set_xticklabels(['Motion', 'Other'])
    ax3.tick_params(axis='both', labelsize=12)
    sns.despine(ax=ax3)

    # plot Edge/RP relationship
    # obtain projection line
    aux_img = mpimg.imread(os.path.join(out_dir, 'aux_fig.png'))
    ax1.imshow(aux_img)
    ax1.axis('off')
    # ax1.tick_params(axis='both', labelsize=12)

    # plot distributions
    # RP
    # sns.histplot(df.loc[df['Motion'] == "True", "RP"],
    #              ax=ax1t,
    #              color=colortrue,
    #              )
    # sns.histplot(df.loc[df['Motion'] == "False", "RP"],
    #              ax=ax1t,
    #              color=colorfalse,
    #              )
    # ax1t.set_xlim([0, 1])

    # # Edge
    # sns.histplot(df.loc[df['Motion'] == "True", "Edge"],
    #              ax=ax1r,
    #              color=colortrue,
    #              kde_kws={'alpha': 0.2}
    #              )
    # sns.histplot(df.loc[df['Motion'] == "False", "Edge"],
    #              ax=ax1r,
    #              color=colorfalse,
    #              )
    # ax1r.set_ylim([0, 1])

    # cosmetics
    for myax in [ax1t, ax1r]:
        myax.set_xticks([])
        myax.set_yticks([])
        myax.set_xlabel('')
        myax.set_ylabel('')
        myax.spines['right'].set_visible(False)
        myax.spines['top'].set_visible(False)
        myax.spines['bottom'].set_visible(False)
        myax.spines['left'].set_visible(False)

    # bring tickmarks back
    for myax in fig.get_axes():
        myax.tick_params(which="major", direction='in', length=3)

    # add figure title
    plt.suptitle('Component Assessment', fontsize=20)

    # outtakes
    plt.savefig(os.path.join(out_dir, 'ICA_AROMA_component_assessment.svg'),
                bbox_inches='tight', dpi=300)

    return
