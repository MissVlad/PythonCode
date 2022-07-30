import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path
def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/results/N2EXMarket/2017to2019_ISGT2022andPeriodPre"
    data = {
        "SSFM1": loadmat(data_file_path/"data_PredictedPrice_SSFM1.mat"),
        "PFM": loadmat(data_file_path / "data_PricePeriodPre_5periodperday.mat"),

    }

    x = np.arange(0, 336, 1.0)
    ax = series(x=np.concatenate([x, [336.]]),
                y=np.concatenate([data["SSFM1"]["act_aveweek"].flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="N2EX Day-ahead electricity price (£/MWh)",
                linestyle="-",
                color="tab:red",
                label="Actual",
                x_ticks=([0, 48, 96, 144, 192, 240, 288, 336],
                         ["Mon.", "Tue.", "Wed.", "Thur.", "Fri.", "Sat.", "Sun.","Mon."]),
                x_lim=(-1,337),
                y_lim=(15, 80),
                linewidth=2.5,
                legend_loc='upper center',
                legend_ncol=1,
                figure_size=(12, 4.3))
    ax = series(x=x,
                y=data["SSFM1"]["pre_aveweek0"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:blue",
                label="SSFM",
                linewidth=2.25,
                legend_loc='upper center',
                legend_ncol=2)
    ax = series(x=x,
                y=data["PFM"]["pre_aveweek"].flatten(),
                ax=ax,
                linestyle="-.",
                color="k",
                label="PFM",
                linewidth=2,
                legend_loc='upper center',
                legend_ncol=3)

if __name__ == "__main__":
        eg_series_uncertainty_plot()