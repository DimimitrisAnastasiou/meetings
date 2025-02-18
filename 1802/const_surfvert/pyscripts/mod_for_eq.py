import argparse
import os
import numpy as np

def read_last_value_from_xvg(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('@')]
    if lines:
        last_line = lines[-1].split()
        return float(last_line[-1])  # Last column of the last row
    return None
def natural_sort_key(s):
    """Extracts the numeric part from 'dtsXX.tsi' filenames for proper sorting."""
    parts = s.replace("dts", "").replace(".tsi", "")  # Remove 'dts' prefix and '.tsi' suffix
    return int(parts)  # Convert the remaining number to integer for correct sorting

def calculate_gamma(ao,nt=1):
    factor = np.sqrt(3 / 4)
    gamma = (ao / (nt * factor)) - 0.5
    return gamma

def main():
    parser = argparse.ArgumentParser(description="Modify equilibrations for productions")
    parser.add_argument('-txt', '--output', required=True, help="Directories file")
    args = parser.parse_args()
    
    directories_file = args.output

    with open(directories_file, 'r') as f:
        directories = [line.strip() for line in f if line.strip()]
    
    for i, directory in enumerate(directories):
        xvg_file_path = os.path.join(directory, "dts-en.xvg")
        if not os.path.exists(xvg_file_path):
            print(f"File not found: {xvg_file_path}")
            continue
        
        ao = read_last_value_from_xvg(xvg_file_path)
        if ao is None:
            print(f"Failed to extract Ao from {xvg_file_path}")
            continue
        
        gamma = calculate_gamma(ao)
        
        traj_file_path = os.path.join(directory, "TrajTSI")
        file = sorted(os.listdir(traj_file_path), key=natural_sort_key)[-1]
        
        copy_path = os.path.join(directory,"input.dts")
        paste_path = os.path.join(directory,"eq","input.dts")
        cmd = f"sed 's/TotalAreaCoupling = HarmonicPotential 0 0.4/TotalAreaCoupling = HarmonicPotential 10000 {gamma:.2f}/' '{copy_path}' > '{paste_path}'"
        os.system(cmd)

        print(i)

if __name__ == "__main__":
    main()
