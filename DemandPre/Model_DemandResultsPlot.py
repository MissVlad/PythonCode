import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path

# %% John MV predictions plot - 2stage1_1 & 2
def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/results/JohnMV/UsedForISGT2022"
    data = {
        "SSFM1": loadmat(data_file_path/"data_PredictedLoad_SSFM1.mat"),
        "SSFM2_2": loadmat(data_file_path/"data_CombinedLoad_SSFM2_variant2.mat"),
        "SSFM2_3": loadmat(data_file_path / "data_CombinedLoad_SSFM2_variant3.mat"),
        "TSFM1_1": loadmat(data_file_path/"data_CombinedLoad_TSFM1_variant1.mat"),
        "TSFM1_2": loadmat(data_file_path / "data_CombinedLoad_TSFM1_variant2.mat"),
        "TSFM1_3": loadmat(data_file_path / "data_CombinedLoad_TSFM1_variant3.mat"),
        "TSFM2_1": loadmat(data_file_path / "data_CombinedLoad_TSFM2_variant1.mat"),
        "TSFM2_2": loadmat(data_file_path / "data_CombinedLoad_TSFM2_variant2.mat"),
        "TSFM2_3": loadmat(data_file_path / "data_CombinedLoad_TSFM2_variant3.mat"),
    }

    x = np.arange(0, 336, 1.0)
    ax = series(x=np.concatenate([x, [336.]]),
                y=np.concatenate([data["SSFM2_2"]["act_minweek2"].flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="Active power demand, AP (MW)",
                linestyle="-",
                color="tab:red",
                label="Actual",
                x_ticks=([0,48,96,144,192,240,288,336],["Mon.", "Tue.", "Wed.", "Thur.", "Fri.","Sat.", "Sun.","Mon."]),
                x_lim=(-1, 337),
                y_ticks=([10, 15, 20, 25, 30],
                        ["10", "15", "20", "25", "30"]),
                y_lim=(10,30),
                linewidth=2.5,
                legend_loc=(1,0),
                legend_ncol=1,
                figure_size=(12, 4.3))
    ax = series(x=x,
                y=data["SSFM1"]["pre_minweek_base"].flatten(),
                ax=ax,
                linestyle=":",
                color="tab:cyan",
                label="SSFM1",
                linewidth=2.35,
                legend_loc=(1,0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["SSFM2_2"]["com_minweek2"].flatten(),
                ax=ax,
                linestyle=":",
                color="tab:olive",
                label="SSFM2(2)",
                linewidth=2.25,
                legend_loc=(1,0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["SSFM2_3"]["com_minweek3"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:pink",
                label="SSFM2(3)",
                linewidth=2.15,
                legend_loc=(1,0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["TSFM1_1"]["com_minweek1"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:purple",
                label="TSFM1(1)",
                linewidth=2,
                legend_loc=(1,0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["TSFM1_2"]["com_minweek2"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:brown",
                label="TSFM1(2)",
                linewidth=1.95,
                legend_loc=(1,0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["TSFM1_3"]["com_minweek3"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:green",
                label="TSFM1(3)",
                linewidth=1.85,
                legend_loc=(1,0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["TSFM2_1"]["com_minweek1"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:blue",
                label="TSFM2(1)",
                linewidth=1.75,
                legend_loc=(1,0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["TSFM2_2"]["com_minweek2"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:orange",
                label="TSFM2(2)",
                linewidth=1.65,
                legend_loc=(1,0),
                legend_ncol=1)
    ax = series(x=x,
                y=data["TSFM2_3"]["com_minweek3"].flatten(),
                ax=ax,
                linestyle="--",
                color="k",
                label="TSFM2(3)",
                linewidth=1.55,
                legend_loc=(1,0),
                legend_ncol=1)
if __name__ == "__main__":
    eg_series_uncertainty_plot()


