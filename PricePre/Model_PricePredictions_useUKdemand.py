import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path

# %% N2EX day-ahead electricity price forecasting 1 - use previous 7 days price to price
# %% N2EX day-ahead electricity price forecasting 2 - use previous 7 days price and previous 7 days demand to price
# %% N2EX day-ahead electricity price forecasting 3 - use previous 7 days price and previous 7 days demand to price /demand

def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/results/N2EXMarket/useDemandAndPricetoPrice"
    data = {
       # "Base1": loadmat(data_file_path/"data_PredictedPrice_SSFM1.mat"),
        "Base2": loadmat(data_file_path/"data_PrePrice_inDandP_P_rightweeks.mat"),
        "Base3": loadmat(data_file_path/"data_PrePrice_inDandP_PandD_rightweeks.mat"),
    }

    x = np.arange(0, 336, 1.0)
    ax = series(x=np.concatenate([x, [336.]]),
                y=np.concatenate([data["Base2"]["act_maxweek"].flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="N2EX day-ahead auction electricity price (GB/MWh)",
                linestyle="-",
                color="tab:red",
                label="Actual price",
                x_ticks=([0, 48, 96, 144, 192, 240, 288, 336],
                         ["Mon.", "Tue.", "Wed.", "Thur.", "Fri.", "Sat.", "Sun.", "Mon."]),
                x_lim=(0, 336),
                y_lim=(20, 75),
                linewidth=2.5,
                legend_loc='upper center',
                legend_ncol=1,
                figure_size=(10, 5))
    ax = series(x=x,
                y=data["Base2"]["pre_maxweek"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:blue",
                label="Base 1",
                linewidth=2.25,
                legend_loc='upper center',
                legend_ncol=2)
    ax = series(x=x,
                y=data["Base3"]["pre_maxweek"].flatten(),
                ax=ax,
                linestyle="-.",
                color="k",
                label="Base 2",
                linewidth=2,
                legend_loc='upper center',
                legend_ncol=3)

if __name__ == "__main__":
    eg_series_uncertainty_plot()