import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def read_simulation_data(txt_folder):
    """
    Reads all the .txt files in the given folder and extracts simulation parameters.

    Args:
        txt_folder (str): Path to the folder containing .txt files.

    Returns:
        pandas.DataFrame: A DataFrame containing Radius, Lambda, Tau, and Status.
    """
    data = []
    for file in os.listdir(txt_folder):
        if file.endswith(".txt") and file.startswith("R"):
            file_path = os.path.join(txt_folder, file)
            with open(file_path, "r") as f:
                params = {}
                for line in f:
                    try:
                        key, value = line.strip().split(": ")
                        params[key] = float(value) if key != "Status" else value.strip()
                    except ValueError:
                        print(f"Skipping malformed line in {file}: {line.strip()}")
                data.append(params)
    return pd.DataFrame(data)


def linear_fit(x, m, b):
    """Linear function for curve fitting."""
    return m * x + b


def plot_lambda_vs_tau_with_fit(data, output_folder):
    """
    Plots Lambda vs Tau for each Radius and adds the fitting information.

    Args:
        data (pandas.DataFrame): The simulation data.
        output_folder (str): Path to save the plots.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(data.info())
    unique_radii = data['Radius'].unique()

    value_dict = {
    2: 2.60,
    8: 9.11,
    3: 3.74,
    10: 11.48,
    4: 4.78,
    6: 7.19
}
    for radius in unique_radii:
        subset = data[data['Radius'] == radius].copy()

        # Encode Status as binary: 0 for Closed, 1 for Opened
        subset['Status'] = subset['Status'].apply(
            lambda x: 1 if x.lower() == 'opened' else (0 if x.lower() == 'closed' else np.nan)
        )

        # Drop rows with invalid or unknown Status values
        subset = subset.dropna(subset=['Status'])
        subset['Status'] = subset['Status'].astype(int)

        # Skip if only one class is present
        # if subset['Status'].nunique() < 2:
        #     print(f"Skipping Radius {radius}: Only one class present or invalid data.")
        #     continue

        # Separate the data into "closed" and "opened"
        closed = subset[subset['Status'] == 0]
        opened = subset[subset['Status'] == 1]

        # Calculate midpoints between transitions
        midpoints = []
        for tau in sorted(subset['Tau'].unique()):
            tau_subset = subset[subset['Tau'] == tau].sort_values(by='Lambda')
            for i in range(1, len(tau_subset) - 1):
                prev_status = tau_subset.iloc[i - 1]['Status']
                current_status = tau_subset.iloc[i]['Status']
                next_status = tau_subset.iloc[i + 1]['Status']

                if prev_status == 1 and current_status == 0 and next_status == 0:
                    midpoint = tau_subset.iloc[i - 1:i + 2]['Lambda'].mean()
                    midpoints.append((tau, midpoint))

        # Convert midpoints to a numpy array
        midpoints = np.array(midpoints)
        if len(midpoints) < 2:
            print(f"Skipping Radius {radius}: Not enough midpoints to fit a line.")
            continue
        
        
        tau_mid = midpoints[:, 0]
        lambda_mid = midpoints[:, 1]

        #Fit a line to the midpoints
        popt, _ = curve_fit(linear_fit, tau_mid, lambda_mid)
        slope, intercept = popt

        # Plot the data
        plt.figure(figsize=(8, 6))
        plt.scatter(closed['Tau'], closed['Lambda'], color='blue', label='Closed', alpha=0.7)
        plt.scatter(opened['Tau'], opened['Lambda'], color='orange', label='Opened', alpha=0.7)
        # plt.scatter(tau_mid, lambda_mid, color='purple', label='Midpoints', zorder=5, edgecolors='black')

        # Plot the fitted line
        tau_range = np.linspace(subset['Tau'].min(), subset['Tau'].max(), 100)
        lambda_fit = linear_fit(tau_range, slope, intercept)
        plt.plot(tau_range, lambda_fit, color='green', linestyle='--', linewidth=1.5,
                 label=f'Fit: slope={slope:.2f}, intercept={intercept:.2f}')

        # Add labels, grid, and legend
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlim((0, subset['Tau'].max() + 0.5))
        plt.ylim((subset['Lambda'].min() - 0.5, subset['Lambda'].max() + 0.5))
        plt.xlabel("$\\tau$ (Tau)", fontsize=12)
        plt.ylabel("$\\lambda$ (Lambda)", fontsize=12)
        plt.title(f" $\\lambda$ vs $\\tau$ (Radius {value_dict[radius]:.2f})", fontsize=14)
        plt.legend(fontsize=10, loc='best')

        # Save the plot
        plot_path = os.path.join(output_folder, f"lambda_vs_tau_radius_{value_dict[radius]:.2f}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Plot saved: {plot_path}")


# Main execution
if __name__ == "__main__":
    # Path to folder with .txt files
    folder_path = "txts"  # Replace with your folder path
    output_folder = f"all_plots_opclo"  # Replace with desired output folder path

    # Load data and plot
    # print(os.listdir(folder_path))
    simulation_data = read_simulation_data(folder_path)
    plot_lambda_vs_tau_with_fit(simulation_data, output_folder)
