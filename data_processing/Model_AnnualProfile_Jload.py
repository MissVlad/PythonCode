import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path

# %% John MV annual profile plot - from 2007 to 2011
def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/processed_data/JohnMVdata"
    data = loadmat(data_file_path/"data_annualprofile_Jload.mat")

    x = np.arange(0, 48, 1.0)
    ax = series(x=np.concatenate([x, [24.]]),
                y=np.concatenate([data["jload_workday_annual_07to11"].flatten(), [np.nan]]),
                x_label="Time of the day",
                y_label="Active power demand, AP (MW)",
                linestyle="-",
                color="tab:blue",
                label="Working day",
                x_ticks=([0, 12, 24, 36, 48], ["00:00", "06:00", "12:00", "18:00", "24:00"]),
                x_lim=(-1.0, 49),
                y_lim=(10, 40),
                legend_loc='upper center',
                legend_ncol=1,
                figure_size=(5, 3))
    ax = series(x=x,
                y=data["jload_weekend_annual_07to11"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:red",
                label="Weekend",
                legend_loc='upper center',
                legend_ncol=2)

if __name__ == "__main__":
    eg_series_uncertainty_plot()
