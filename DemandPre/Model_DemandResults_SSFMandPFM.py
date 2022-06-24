import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path

# %% John MV predictions plot - 2stage1_1 & 2
def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/results/JohnMV/UsedForISGT2022andPeriodPre"
    data = {
        "SSFM1": loadmat(data_file_path/"data_PredictedLoad_SSFM1.mat"),
        "PFM1": loadmat(data_file_path/"data_PredictedLoad_PFM1.mat"),
    }

    x = np.arange(0, 336, 1.0)
    ax = series(x=np.concatenate([x, [336.]]),
                y=np.concatenate([data["PFM1"]["act_maxweek2"].flatten(), [np.nan]]),
                x_label="Day of the week",
                y_label="Active power demand, AP (MW)",
                linestyle="-",
                color="tab:red",
                label="Actual",
                x_ticks=([0,48,96,144,192,240,288,336],["Mon.", "Tue.", "Wed.", "Thur.", "Fri.","Sat.", "Sun.","Mon."]),
                x_lim=(-1, 337),
                #y_ticks=([10, 15, 20, 25, 30],
                        #["10", "15", "20", "25", "30"]),
                y_lim=(20,50),
                linewidth=2.5,
                legend_loc='upper center',
                legend_ncol=1,
                figure_size=(12, 4.3))
    ax = series(x=x,
                y=data["SSFM1"]["pre_maxweek_base"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:blue",
                label="SSFM1",
                linewidth=2.25,
                legend_loc='upper center',
                legend_ncol=2)
    ax = series(x=x,
                y=data["PFM1"]["pre_maxweek_PF1"].flatten(),
                ax=ax,
                linestyle="-.",
                color="k",
                label="PFM1",
                linewidth=2,
                legend_loc='upper center',
                legend_ncol=3)
if __name__ == "__main__":
    eg_series_uncertainty_plot()