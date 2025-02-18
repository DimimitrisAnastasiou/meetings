import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.image as mpimg
from matplotlib.cm import get_cmap

def read_simulation_data(folders):
    """
    Reads all the .txt files from multiple folders and extracts simulation parameters.

    Args:
        folders (list): List of folder paths containing txt files.

    Returns:
        pandas.DataFrame: A DataFrame containing Radius, Lambda, Tau, Status, and other parameters.
    """
    data = []
    processed_files = 0
    skipped_files = 0
    for folder in folders:
        print(f"READING FOLDER: {folder}")
        print(f"Number of files in folder: {len(os.listdir(folder))}")
        for file in os.listdir(folder):
            if file.endswith(".txt") and file.startswith("R"):
                file_path = os.path.join(folder, file)
                try:
                    with open(file_path, "r") as f:
                        params = {}
                        for line in f:
                            try:
                                key, value = line.strip().split(": ")
                                params[key] = float(value) if key != "Status" else value.strip()
                            except ValueError as e:
                                print(f"Error parsing line in {file_path}: {line.strip()} - {e}")
                                continue
                        required_keys = {"Radius", "Lambda", "Tau", "Status"}
                        if not required_keys.issubset(params.keys()):
                            print(f"Skipping file {file_path}: Missing keys {required_keys - params.keys()}")
                            skipped_files += 1
                            continue
                        data.append(params)
                        processed_files += 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    skipped_files += 1
    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")
    return pd.DataFrame(data)

def linear_fit(x, m, b):
    """Linear function for curve fitting."""
    return m * x + b

def calculate_midpoints(subset, x_col, y_col):
    """
    Calculate midpoints between Closed and Opened transitions.

    Args:
        subset (DataFrame): Subset of the data.
        x_col (str): The column to group by (Tau or Radius).
        y_col (str): The column to calculate midpoints (Lambda or Radius).

    Returns:
        list: A list of midpoints as (x, y) tuples.
    """
    midpoints = []
    grouped = subset.groupby(x_col)  # Group by Tau or Radius
    for x, group in grouped:
        # Sort the group by y_col (Lambda or Radius)
        group = group.sort_values(by=y_col)

        # Process the group to identify transitions
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
    n_colors = len(unique_groups)
    cmap = plt.get_cmap("viridis")  # Change to "viridis" or "plasma" if desired
    colors = [cmap(i / n_colors) for i in range(n_colors)]  # Generate distinct colors
    
    # Specify the path to your TXT file
    file_path = "info_surf.txt"

    # Read the TXT file as a CSV (comma-separated values)
    df = pd.read_csv(file_path, sep=',')

    # Display the first few rows of the DataFrame
    print(df.head())

    mean = df["Calc_Radius"].mean()
    print("MEAN RADIUS:",mean)

    print("HEHE:",len(unique_groups))
    plt.figure(figsize=(8, 6))

    for group, color in zip(unique_groups, colors):
        subset = data[data[group_col] == group].copy()

        # Encode Status as binary: 0 for Closed, 1 for Opened
        subset['Status'] = subset['Status'].apply(lambda x: 1 if x.lower() == 'opened' else (0 if x.lower() == 'closed' else np.nan))

        # Drop rows with invalid or unknown Status values
        subset = subset.dropna(subset=['Status'])
        subset['Status'] = subset['Status'].astype(int)

        # Check if there are at least two classes in the data
        if subset['Status'].nunique() < 2:
            print(f"Skipping {group_col} {group}: Only one class present or invalid data.")
            continue

        midpoints = calculate_midpoints(subset, x_col, y_col)
        midpoints = np.array(midpoints)
        print(len(midpoints))

        if len(midpoints) < 2:
            print(f"Skipping {group_col} {group}: Not enough midpoints to fit a line.")
            continue

        # Calculate averages of midpoints for each x_col
        avg_midpoints = {}
        for x, y in midpoints:
            if x not in avg_midpoints:
                avg_midpoints[x] = []
            avg_midpoints[x].append(y)

        x_mid = np.array(sorted(avg_midpoints.keys())) 
        y_mid = np.array([np.mean(avg_midpoints[x]) for x in x_mid])
        # errors = np.array([np.std(avg_midpoints[x]) for x in x_mid_R])

        # Fit a line to the average midpoints using errors as weights
        popt, pcov = curve_fit(linear_fit, x_mid, y_mid)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))

        # Extract the radius value
        radius_value = subset['Radius'].iloc[0] if 'Radius' in subset.columns else 'N/A'

        if radius_value % 0.5 == 0 and not radius_value.is_integer() or radius_value==5 or radius_value==7 or radius_value>8:
            continue

        # Plot the fitted line
        x_range = np.linspace(0, np.max(x_mid), 100)  # Adjust range as needed
        y_fit = linear_fit(x_range, slope, intercept)
        plt.plot(
            x_range, y_fit, color=color, linestyle='--', linewidth=1.0,
            label=f'R:{mean:.2f}~{radius_value} | Fit: sl={slope:.2f}±{slope_err:.2f}, int={intercept:.2f}±{intercept_err:.2f}'
        )

        # Plot the average midpoints with error bars
        plt.scatter(x_mid, y_mid,  color=color)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel(f"{x_col}", fontsize=12)
    plt.ylabel(f"{y_col}", fontsize=12)
    # plt.ylim((3, 10))  # Adjust as needed
    plt.title(f"No proteins: {y_col} vs {x_col}", fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_folder, f"{y_col}_vs_{x_col}_{group_col}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot saved: {plot_path}")

# Example usage
folders = ["txts_from_directories_surf_prod"]
# output_folder_lambda = "plots_with_fit_Lambda_Erros_all_new"
output_folder_lambda = f"plot_surf"

data = read_simulation_data(folders)
plot_with_fit(data, 'Radius', 'Tau', 'Lambda', output_folder_lambda)