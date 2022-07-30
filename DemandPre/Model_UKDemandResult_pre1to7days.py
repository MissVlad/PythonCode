import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path

# %% UK Demand predictions plot - use time information and previous 1 to 7 days' demand as input
def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/results/UKDemand"
    data = {
        "pre1": loadmat(data_file_path/"data_UKDemandPreResult_preday_1.mat"),
        "pre2": loadmat(data_file_path/"data_UKDemandPreResult_preday_2.mat"),
        "pre3": loadmat(data_file_path / "data_UKDemandPreResult_preday_3.mat"),
        "pre4": loadmat(data_file_path/"data_UKDemandPreResult_preday_4.mat"),
        "pre5": loadmat(data_file_path / "data_UKDemandPreResult_preday_5.mat"),
        "pre6": loadmat(data_file_path / "data_UKDemandPreResult_preday_6.mat"),
        "pre7": loadmat(data_file_path / "data_UKDemandPreResult_preday_7.mat"),
    }

    x = np.arange(0, 336, 1.0)
    ax = series(x=np.concatenate([x, [336.]]),
                y=np.concatenate([data["pre1"]["act_aveweek"].flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="UK demand (MW)",
                linestyle="-",
                color="tab:red",
                label="Actual",
                x_ticks=([0, 48, 96, 144, 192, 240, 288, 336],
                         ["Mon.", "Tue.", "Wed.", "Thur.", "Fri.", "Sat.", "Sun.", "Mon."]),
                x_lim=(-1, 337),
                # y_ticks=([10, 15, 20, 25, 30],
                        # ["10", "15", "20", "25", "30"]),
               # y_lim=(10, 30),
                linewidth=2.5,
                legend_loc=(1, 0),
                legend_ncol=1,
                figure_size=(12, 4.3))
    ax = series(x=x,
                y=data["pre1"]["pre_aveweek"].flatten(),
                ax=ax,
                linestyle="-",
                color="tab:cyan",
                label="Forecast 1",
                linewidth=2.35,
                legend_loc=(1, 0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["pre2"]["pre_aveweek"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:olive",
                label="Forecast 2",
                linewidth=2.25,
                legend_loc=(1, 0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["pre3"]["pre_aveweek"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:pink",
                label="Forecast 3",
                linewidth=2.15,
                legend_loc=(1, 0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["pre4"]["pre_aveweek"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:purple",
                label="Forecast 4",
                linewidth=2,
                legend_loc=(1, 0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["pre5"]["pre_aveweek"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:brown",
                label="Forecast 5",
                linewidth=1.95,
                legend_loc=(1, 0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["pre6"]["pre_aveweek"].flatten(),
                ax=ax,
                linestyle="-",
                color="tab:green",
                label="Forecast 6",
                linewidth=1.85,
                legend_loc=(1, 0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["pre7"]["pre_aveweek"].flatten(),
                ax=ax,
                linestyle="-",
                color="k",
                label="Forecast 7",
                linewidth=1.75,
                legend_loc=(1, 0),
                legend_ncol=1)
if __name__ == "__main__":
    eg_series_uncertainty_plot()