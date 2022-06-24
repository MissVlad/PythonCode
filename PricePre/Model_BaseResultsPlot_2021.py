import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path
def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/results/N2EXMarket/2017to2021"
    data = {
        "Base1": loadmat(data_file_path/"data_Base1Results_2021_originaldata.mat"),
        "Base2": loadmat(data_file_path/"data_Base2Results_2021_originaldata.mat"),
        "Base3": loadmat(data_file_path/"data_Base3Results_2021_originaldata.mat"),
        "Base4": loadmat(data_file_path/"data_Base4Results_2021_originaldata.mat"),
    }

    x = np.arange(0, 336, 1.0)
    ax = series(x=np.concatenate([x, [336.]]),
                y=np.concatenate([data["Base1"]["act_price_maxweek_Base1"].flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="N2EX day-ahead auction electricity price (GB/MWh)",
                linestyle="--",
                color="tab:red",
                label="Actual price",
                x_ticks=([0, 48, 96, 144, 192, 240, 288, 336],
                         ["Mon.", "Tue.", "Wed.", "Thur.", "Fri.", "Sat.", "Sun.", "Mon."]),
                x_lim=(0, 336),
                y_lim=(-20, 2800),
                linewidth=2.5,
                legend_loc='upper center',
                legend_ncol=1,
                figure_size=(10, 5))
    ax = series(x=x,
                y=data["Base1"]["pre_price_maxweek_Base1"].flatten(),
                ax=ax,
                linestyle=":",
                color="tab:blue",
                label="Base 1",
                linewidth=2.25,
                legend_loc='upper center',
                legend_ncol=2)
    ax = series(x=x,
                y=data["Base2"]["pre_price_maxweek_Base2"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:green",
                label="Base 2",
                linewidth=2,
                legend_loc='upper center',
                legend_ncol=3)
    ax = series(x=x,
                y=data["Base3"]["pre_price_maxweek_Base3"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:orange",
                label="Base 3",
                linewidth=1.75,
                legend_loc='upper center',
                legend_ncol=4)
    ax = series(x=x,
                y=data["Base4"]["pre_price_maxweek_Base4"].flatten(),
                ax=ax,
                linestyle="-",
                color="k",
                label="Base 4",
                linewidth=1.5,
                legend_loc='upper center',
                legend_ncol=5)
if __name__ == "__main__":
    eg_series_uncertainty_plot()