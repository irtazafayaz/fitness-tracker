import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 2]
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    label_df = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(label_df["acc_y"].reset_index(drop=True), label=label)
    plt.title(f"Accelerometer Y-axis - {label}")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.show()

for label in df["label"].unique():
    label_df = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(label_df[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.title(f"Accelerometer Y-axis - {label}")
    plt.xlabel("Time")
    plt.ylabel("Acceleration (g)")
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query(
    "participant == 'A'").reset_index(drop=True)
# category_df.groupby("category").plot(y="acc_y")
category_df.groupby(["category"])["acc_y"].plot()
ax.set_xlabel("Samples")
ax.set_ylabel("Acceleration (g)")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'squat'").sort_values(
    "participant").reset_index(drop=True)
participant_df.groupby(["participant"])["acc_y"].plot()
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = 'squat'
participant = 'A'
all_axis_df = df.query(f"label == '{label}'").query(
    f"participant == '{participant}'").reset_index(drop=True)

fig, ax = plt.subplots()
# all_axis_df.plot(y=["acc_x", "acc_y", "acc_z"])
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_xlabel("Samples")
ax.set_ylabel("Acceleration (g)")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:

        subset_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )
        if subset_df.empty:
            continue
        fig, ax = plt.subplots()
        subset_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
        ax.set_title(f"Participant: {participant} - Exercise: {label}")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Acceleration (g)")
        plt.legend()
        plt.show()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20, 30))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].set_ylabel("Acceleration (g)")
ax[0].set_title(f"Participant: {participant} - Exercise: {label}")
ax[1].legend(title="Gyroscope")
ax[1].set_ylabel("Angular Velocity (deg/s)")
ax[1].set_xlabel("Samples")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:

        subset_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )
        if subset_df.empty:
            continue
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
        subset_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
        subset_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

        ax[0].set_ylabel("Acceleration (g)")
        ax[0].set_title(f"Participant: {participant} - Exercise: {label}")
        ax[1].legend(title="Gyroscope")
        ax[1].set_ylabel("Angular Velocity (deg/s)")
        ax[1].set_xlabel("Samples")
        plt.legend()
        plt.savefig(f"../../reports/figures/{participant}_{label}_sensors.png")
        plt.show()
