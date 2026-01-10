from signal import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_data_outliers_removed.pkl")
predictor_columns = df.columns[:6].tolist()

# Plot settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()
subset = df[df["set"] == 35]["gyr_x"].plot()

for col in predictor_columns:
    df[col].interpolate(method='linear', limit_direction='both', inplace=True)

# ------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 4]["acc_y"].plot()
df[df["set"] == 4]

type(df["set"])
type(df)
type(df[df["set"] == 25])
type(df["set"] == 25)

start_time = df[df["set"] == 1].index[0]
end_time = df[df["set"] == 1].index[-1]
duration = end_time - start_time

for set_name in df["set"].unique():
    start_time = df[df["set"] == set_name].index[0]
    end_time = df[df["set"] == set_name].index[-1]
    duration = (end_time - start_time).total_seconds()
    df.loc[df["set"] == set_name, "duration"] = duration

df.head()

duration_df = df.groupby("category")["duration"].mean()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

time_diff = (df.index[1] - df.index[0]).total_seconds()
fs = 1 / time_diff
cutoff = 1.2

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(
    drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(
    0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(
    0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)

    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

pca_df = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(pca_df, predictor_columns)

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(pc_values) + 1), pc_values)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance Ratio")
plt.xticks(range(1, len(pc_values) + 1))
plt.show()

# From the figure we can see that the first 3 components explain most of the variance.
pca_df = PCA.apply_pca(pca_df, predictor_columns, number_comp=3)

subset = pca_df[pca_df["set"] == 45]

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = pca_df.copy()
acc_r = df_squared["acc_x"]**2 + \
    df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"]**2 + \
    df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2
df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws = int(1000 / 200)
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")
df_temporal

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
