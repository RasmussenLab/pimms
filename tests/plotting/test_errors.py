from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.testing.decorators import image_comparison
from pimmslearn.plotting.errors import (get_data_for_errors_by_median,
                                        plot_errors_by_median)

TOP_N_COLOR_PALETTE = {'TRKNN': (0.20125317221201128, 0.6907920815379025, 0.47966761189275336),
                       'KNN': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                       'RF': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                       'KNN_IMPUTE': (1.0, 0.4980392156862745, 0.054901960784313725),
                       'SEQKNN': (0.7632105624545802, 0.5838460616396939, 0.19465686802007026)}

file_dir = Path(__file__).resolve().parent


@pytest.fixture
def example_data():
    """
    Fixture to load example data from a csv file for testing.
    """
    example_data_path = file_dir / 'pred_test.csv'
    return pd.read_csv(example_data_path, index_col=[0, 1])


@pytest.fixture
def feat_medians():
    medians_path = file_dir / 'test_medians.csv'
    s = pd.read_csv(medians_path, index_col=0).squeeze()
    return s


# @pytest.fixture
def expected_errors_binned():
    errors_binned_path = file_dir / 'exp_errors_binned.csv'
    df = pd.read_csv(errors_binned_path, sep=',', index_col=0)
    df = df.astype({'intensity binned by median of Gene Names': 'category'})
    return df


@pytest.fixture
def expected_plotted():
    plotted_path = file_dir / 'expected_plotted.csv'
    df = pd.read_csv(plotted_path, sep=',', index_col=0)
    return df


def test_get_data_for_errors_by_median(expected_plotted):
    expected_errors_binned_ = expected_errors_binned()
    plotted = get_data_for_errors_by_median(
        errors=expected_errors_binned_,
        feat_name='Gene Names',
        metric_name='MAE',
        seed=42,
    )

    pd.testing.assert_frame_equal(plotted, expected_plotted)


# @image_comparison(baseline_images=['errors_by_median'], remove_text=True,
#                   extensions=['png'], style='mpl20')
def test_plot_errors_by_median(example_data, feat_medians):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax, errors_binned = plot_errors_by_median(
        example_data,
        feat_medians=feat_medians,
        ax=ax,
        feat_name='Gene Names',
        palette=TOP_N_COLOR_PALETTE,
        metric_name='MAE',)
    ax.set_ylabel("Average error (MAE)")
    ax.legend(loc='best', ncols=5)
    fig.tight_layout()

    expected_errors_binned_ = expected_errors_binned()
    pd.testing.assert_frame_equal(errors_binned, expected_errors_binned_)
