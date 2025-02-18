import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

def linear_fit(x, m, b):
    """Linear function for curve fitting."""
    return m * x + b

def calculate_midpoints(data, x_col, y_col):
    """
    Calculate midpoints between Closed and Opened transitions.

    Args:
        data (DataFrame): Processed DataFrame.
        x_col (str): Column to group by (Tau or Radius).
        y_col (str): Column for midpoints (Lambda or Calc_Radius).

    Returns:
        list: Midpoints as (x, y) tuples.
    """
    midpoints = []
    grouped = data.groupby(x_col)
    for x, group in grouped:
        group = group.sort_values(by=y_col)
        for i in range(1, len(group) - 1):
            prev_status = group.iloc[i - 1]['Status']
            current_status = group.iloc[i]['Status']
            next_status = group.iloc[i + 1]['Status']
            if prev_status != current_status and next_status == current_status:
                midpoint = group.iloc[i - 1:i + 2][y_col].mean()
                midpoints.append((x, midpoint))
    return midpoints

def plot_with_fit(data, group_col, x_col, y_col, output_folder):
    print(data.info())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    unique_groups = data[group_col].unique()
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / len(unique_groups)) for i in range(len(unique_groups))]
    
    mean_radius = data["Calc_Radius"].mean()
    print("MEAN RADIUS:", mean_radius)
    
    plt.figure(figsize=(8, 6))
    for group, color in zip(unique_groups, colors):
        subset = data[data[group_col] == group].copy()
        subset['Status'] = subset['Status'].apply(lambda x: 1 if x.lower() == 'opened' else 0)
        
        midpoints = calculate_midpoints(subset, x_col, y_col)
        midpoints = np.array(midpoints)
        if len(midpoints) < 2:
            print(f"Skipping {group_col} {group}: Not enough midpoints to fit a line.")
            continue

        x_mid = np.array(sorted(set(x for x, _ in midpoints)))
        y_mid = np.array([np.mean([y for xx, y in midpoints if xx == x]) for x in x_mid])
        
        popt, pcov = curve_fit(linear_fit, x_mid, y_mid)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
        
        radius_value = subset['Radius'].iloc[0]
        if radius_value % 0.5 == 0 and not radius_value.is_integer() or radius_value == 5 or radius_value == 7 or radius_value > 8:
            continue
        
        x_range = np.linspace(0, np.max(x_mid), 100)
        y_fit = linear_fit(x_range, slope, intercept)
        plt.plot(x_range, y_fit, color=color, linestyle='--', linewidth=1.0,
                 label=f'R:{mean_radius:.2f}~{radius_value} | Fit: sl={slope:.2f}±{slope_err:.2f}, int={intercept:.2f}±{intercept_err:.2f}')
        plt.scatter(x_mid, y_mid, color=color)
    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel(f"{x_col}", fontsize=12)
    plt.ylabel(f"{y_col}", fontsize=12)
    plt.title(f"No proteins: {y_col} vs {x_col}", fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    plot_path = os.path.join(output_folder, f"{y_col}_vs_{x_col}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved: {plot_path}")

def plot_with_fit_squared(data, group_col, x_col, y_col, output_folder):
    print(data.info())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    unique_groups = data[group_col].unique()
    n_colors = len(unique_groups)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / n_colors) for i in range(n_colors)]

    mean_radius = data["Calc_Radius"].mean()
    print("MEAN RADIUS:", mean_radius)

    plt.figure(figsize=(8, 6))

    for group, color in zip(unique_groups, colors):
        subset = data[data[group_col] == group].copy()

        subset['Status'] = subset['Status'].apply(lambda x: 1 if x.lower() == 'opened' else (0 if x.lower() == 'closed' else np.nan))
        subset = subset.dropna(subset=['Status'])
        subset['Status'] = subset['Status'].astype(int)

        if subset['Status'].nunique() < 2:
            print(f"Skipping {group_col} {group}: Only one class present or invalid data.")
            continue

        midpoints = calculate_midpoints(subset, x_col, y_col)
        midpoints = np.array(midpoints)

        if len(midpoints) < 2:
            print(f"Skipping {group_col} {group}: Not enough midpoints to fit a line.")
            continue

        avg_midpoints = {}
        for x, y in midpoints:
            if x not in avg_midpoints:
                avg_midpoints[x] = []
            avg_midpoints[x].append(y)

        x_mid = np.array(sorted(avg_midpoints.keys()))
        y_mid = np.array([np.mean(avg_midpoints[x]) for x in x_mid])

        x_mid_scaled = x_mid * (subset["Calc_Radius"].iloc[0] ** 2)
        y_mid_scaled = y_mid * subset["Calc_Radius"].iloc[0]

        popt, pcov = curve_fit(linear_fit, x_mid_scaled, y_mid_scaled)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))

        radius_value = subset['Radius'].iloc[0] if 'Radius' in subset.columns else 'N/A'

        if radius_value % 0.5 == 0 and not radius_value.is_integer() or radius_value == 5 or radius_value == 7 or radius_value > 8:
            continue

        x_range = np.linspace(0, np.max(x_mid_scaled), 100)
        y_fit = linear_fit(x_range, slope, intercept)
        plt.plot(
            x_range, y_fit, color=color, linestyle='--', linewidth=1.0,
            label=f'R:{mean_radius:.2f}~{radius_value} | Fit: sl={slope:.2f}±{slope_err:.2f}, int={intercept:.2f}±{intercept_err:.2f}'
        )

        plt.scatter(x_mid_scaled, y_mid_scaled, color=color)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel(f"{x_col} * R^2", fontsize=12)
    plt.ylabel(f"{y_col} * R", fontsize=12)
    plt.title(f"No proteins: {y_col} * R vs {x_col} * R^2", fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    plot_path = os.path.join(output_folder, f"{y_col}_vs_{x_col}_squared.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved: {plot_path}")


# Load data from the consolidated TXT file
data_file = "info_vert.txt"
data = pd.read_csv(data_file)

# Ensure Status is correctly formatted as string
data['Status'] = data['Status'].astype(str)

# Define output directory and plot
output_folder_lambda = "plot_squared_vert_new"
plot_with_fit(data, 'Radius', 'Tau', 'Lambda', output_folder_lambda)
plot_with_fit_squared(data, 'Radius', 'Tau', 'Lambda', output_folder_lambda)