import numpy as np

from Code.Ploting.fast_plot_Func import *
from scipy.io import loadmat
from Code.project_utils import project_path

# %% John MV daily profile within different types of days - original and variants
def eg_series_uncertainty_plot():
    data_file_path = project_path / "Data/processed_data/JohnMVdata"
    data = loadmat(data_file_path/"data_LoadDailyProfile_variant3.mat")

    x = np.arange(0, 24, 0.5)
    ax = series(x=np.concatenate([x, [24.]]),
                y=np.concatenate([data["mon"].flatten(), [np.nan]]),
                x_label="Time of the day",
                y_label="Active power demand, AP (MW)",
                linestyle="-",
                color="tab:red",
                label="Mon.",
                x_ticks=([0, 6, 12, 18, 24], ["00:00", "06:00", "12:00", "18:00", "24:00"]),
                x_lim=(-0.5, 24.5),
                y_lim=(15,45),
                legend_loc='upper center',
                legend_ncol=1,
                figure_size=(5,3))
    ax = series(x=x,
                y=data["fri"].flatten(),
                ax=ax,
                linestyle="-",
                color="tab:blue",
                label="Fri.",
                legend_loc='upper center',
                legend_ncol=2)
    ax = series(x=x,
                y=data["tue"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:orange",
                label="Tue.",
                legend_loc='upper center',
                legend_ncol=3)
    ax = series(x=x,
                y=data["sat"].flatten(),
                ax=ax,
                linestyle="--",
                color="tab:green",
                label="Sat.",
                legend_loc='upper center',
                legend_ncol=4)
    ax = series(x=x,
                y=data["wed"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:brown",
                label="Wed.",
                legend_loc='upper center',
                legend_ncol=1)
    ax = series(x=x,
                y=data["sun"].flatten(),
                ax=ax,
                linestyle="-.",
                color="tab:pink",
                label="Sun.",
                legend_loc='upper center',
                legend_ncol=2)
    ax = series(x=x,
                y=data["thur"].flatten(),
                ax=ax,
                linestyle=":",
                color="black",
                label="Thur.",
                legend_loc='upper center',
                legend_ncol=3)
    ax = series(x=x,
                y=data["hol"].flatten(),
                ax=ax,
                linestyle=":",
                color="tab:purple",
                label="HOL.",
                legend_loc='upper center',
                legend_ncol=4)


if __name__ == "__main__":
    eg_series_uncertainty_plot()