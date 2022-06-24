import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path
def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/results/N2EXMarket/2017to2019_ISGT2022andPeriodPre"
    data = {
        "SSFM1": loadmat(data_file_path/"data_PredictedPrice_SSFM1.mat"),
        "PFM1": loadmat(data_file_path / "data_PredictedPrice_PeriodPre1.mat"),
        "PFM2": loadmat(data_file_path / "data_PredictedPrice_PeriodPre2.mat"),
    }

    x = np.arange(0, 336, 1.0)
    ax = series(x=np.concatenate([x, [336.]]),
                y=np.concatenate([data["SSFM1"]["act_minweek"].flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="N2EX Day-ahead electricity price (£/MWh)",
                linestyle="-",
                color="tab:red",
                label="Actual",
                x_ticks=([0, 48, 96, 144, 192, 240, 288, 336],
                         ["Mon.", "Tue.", "Wed.", "Thur.", "Fri.", "Sat.", "Sun.","Mon."]),
                x_lim=(-1,337),
                y_lim=(0, 85),
                linewidth=2.5,
                legend_loc='upper center',
                legend_ncol=1,
                figure_size=(12, 4.3))
    ax = series(x=x,
                y=data["SSFM1"]["pre_minweek0"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:blue",
                label="SSFM1",
                linewidth=2.25,
                legend_loc='upper center',
                legend_ncol=2)
    ax = series(x=x,
                y=data["PFM1"]["pre_minweek_PF1"].flatten(),
                ax=ax,
                linestyle="-.",
                color="k",
                label="PFM1",
                linewidth=2,
                legend_loc='upper center',
                legend_ncol=3)
    ax = series(x=x,
                y=data["PFM2"]["pre_minweek_PF2"].flatten(),
                ax=ax,
                linestyle="-",
                color="tab:green",
                label="PFM2",
                linewidth=1.5,
                legend_loc='upper center',
                legend_ncol=4)

if __name__ == "__main__":
        eg_series_uncertainty_plot()