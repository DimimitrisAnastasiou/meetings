### get txts with radius 

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import argparse

class TSIFile:
    def __init__(self, file_path):
        """
        Initialize the TSIFile class by loading the .tsi file and extracting system properties.
        """
        self.file_path = file_path
        self.box_size = None
        self.vertex_count = None
        self.triangle_count = None
        self.vertices = []
        self.triangles = []
        self.inclusion_count = None
        self.inclusions = []

        # Load the file content
        self.read_file(self.file_path)

    def get_box_size(self):
        return self.box_size

    def get_vertex_count(self):
        return self.vertex_count

    def get_triangle_count(self):
        return self.triangle_count

    def get_vertices(self):
        return self.vertices

    def get_triangles(self):
        return self.triangles

    def get_vertex_by_index(self, index):
        for vertex in self.vertices:
            if vertex['index'] == index:
                return vertex
        return None

    def validate_triangle_connectivity(self):
        """
        Validates the connectivity of triangles to ensure each edge has a corresponding mirror.
        """
        edge_to_triangle = defaultdict(list)

        # Register each edge of the triangle
        for triangle in self.triangles:
            edges = [
                (triangle['v1'], triangle['v2']),
                (triangle['v2'], triangle['v3']),
                (triangle['v3'], triangle['v1'])
            ]
            for edge in edges:
                # Sort edges to handle undirected links (v1, v2) == (v2, v1)
                edge = tuple(sorted(edge))
                edge_to_triangle[edge].append(triangle['index'])

        # Check that each edge has exactly two references (a "mirror" pair)
        broken_links = []
        for edge, triangles in edge_to_triangle.items():
            if len(triangles) != 2:  # If not mirrored
                broken_links.append(edge)

        # if broken_links:
        #     print("Warning: Broken links detected in the following edges:", broken_links)
        # else:
        #     print("Triangle connectivity is valid.")

        return broken_links

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # BOX SIZE
        self.box_size = [
            float(lines[1].strip().split()[1]),
            float(lines[1].strip().split()[2]),
            float(lines[1].strip().split()[3])
        ]

        # VERTEX
        self.vertex_count = int(lines[2].strip().split()[1])
        for i in range(3, 3 + self.vertex_count):
            parts = lines[i].split()
            vertex = {
                'index': int(parts[0]),
                'x': float(parts[1]),
                'y': float(parts[2]),
                'z': float(parts[3])
            }
            self.vertices.append(vertex)

        # TRIANGLES
        self.triangle_count = int(lines[3 + self.vertex_count].strip().split()[1])
        for i in range(4 + self.vertex_count, 4 + self.vertex_count + self.triangle_count):
            parts = lines[i].split()
            triangle = {
                'index': int(parts[0]),
                'v1': int(parts[1]),
                'v2': int(parts[2]),
                'v3': int(parts[3]),
            }
            self.triangles.append(triangle)

    def calc_radius(self):
        """
        Calculate the radius of the hole by summing up the length of the links and dividing by pi.
        """
        total_length = 0.0
        edge_to_triangle = defaultdict(list)

        # Register each edge of the triangle
        for triangle in self.triangles:
            edges = [
                (triangle['v1'], triangle['v2']),
                (triangle['v2'], triangle['v3']),
                (triangle['v3'], triangle['v1'])
            ]
            for edge in edges:
                edge = tuple(sorted(edge))
                edge_to_triangle[edge].append(triangle['index'])

        # Calculate the total length of edges with no mirrors (hole edges)
        for edge, triangles in edge_to_triangle.items():
            if len(triangles) == 1:  # Edge with no mirror
                v1 = self.get_vertex_by_index(edge[0])
                v2 = self.get_vertex_by_index(edge[1])
                length = math.sqrt((v1['x'] - v2['x'])**2 + (v1['y'] - v2['y'])**2 + (v1['z'] - v2['z'])**2)
                total_length += length

        radius = total_length / (2*math.pi)
        return radius
    
    def visualize_links(self,path,radius):
        """
        Visualize the triangles and highlight broken links in 2D (XY-plane).
        """
        plt.figure(figsize=(10, 10))

        # Plot all triangles
        for triangle in self.triangles:
            vertices = [
                self.get_vertex_by_index(triangle['v1']),
                self.get_vertex_by_index(triangle['v2']),
                self.get_vertex_by_index(triangle['v3']),
                self.get_vertex_by_index(triangle['v1'])  # Close the triangle
            ]
            x_coords = [v['x'] for v in vertices]
            y_coords = [v['y'] for v in vertices]
            # plt.plot(x_coords, y_coords, color='gray', alpha=0.5)

        # Highlight broken links
        broken_links = self.validate_triangle_connectivity()
        for edge in broken_links:
            v1 = self.get_vertex_by_index(edge[0])
            v2 = self.get_vertex_by_index(edge[1])
            plt.plot([v1['x'], v2['x']], [v1['y'], v2['y']], color='red', linewidth=2)

        # Scatter plot for vertices
        for vertex in self.vertices:
            plt.scatter(vertex['x'], vertex['y'], color='blue', s=10)
        
        center_x, center_y = self.box_size[0]/2,self.box_size[1]/2

        # Define the length of the arrow
        arrow_length = radius  # Arrow extends along the x-axis

        # Plot the arrow
        plt.arrow(center_x, center_y, arrow_length, 0,  # Only in x-direction
                head_width=0.3, head_length=0.5, fc='blue', ec='blue')

        plt.xlabel("X-axis")
        plt.xticks(range(0, 51, 3))
        plt.ylabel("Y-axis")
        plt.title(f"calculated radius {radius}")
        plt.grid()
        plt.axis('equal')
        plt.savefig(path)
        print(f"Plot saved at: {path}")
        plt.close()

def process_tsi_files(directory):
    """Processes the first 5 and last 5 TSI files in a directory."""
    traj_tsi_folder = os.path.join(directory, "TrajTSI")
    if not os.path.exists(traj_tsi_folder):
        print(f"TrajTSI folder not found in {directory}, skipping.")
        return None

    tsi_files = sorted([f for f in os.listdir(traj_tsi_folder) if f.startswith("dts") and f.endswith(".tsi")])

    if len(tsi_files) < 10:
        print(f"Not enough TSI files in {traj_tsi_folder}, skipping.")
        return None

    first_five = tsi_files[:5]
    last_five = tsi_files[-5:]

    first_radii = []
    last_radii = []

    for tsi_file in first_five:
        file_path = os.path.join(traj_tsi_folder, tsi_file)
        tsi = TSIFile(file_path)
        first_radii.append(len(tsi.validate_triangle_connectivity()))

    for tsi_file in last_five:
        file_path = os.path.join(traj_tsi_folder, tsi_file)
        tsi = TSIFile(file_path)
        last_radii.append(len(tsi.validate_triangle_connectivity()))

    first_avg = np.mean(first_radii)
    last_avg = np.mean(last_radii)

    status = "Opened" if last_avg > first_avg else "Closed"
    return first_avg, last_avg, status

def write_simulation_info(output_folder, radius, lambda_val, tau, status):
    """Writes the simulation information in the specified structure."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = f"R{radius:.2f}_L{lambda_val}_T{tau}.txt"
    output_path = os.path.join(output_folder, file_name)

    with open(output_path, 'w') as f:
        f.write(f"Radius: {radius:.2f}\n")
        f.write(f"Lambda: {lambda_val:.2f}\n")
        f.write(f"Tau: {tau:.2f}\n")
        f.write(f"Status: {status}\n")
    print(f"Simulation info saved to {output_path}")

def fit_circle(x, y):
    """
    Fits a circle to given x, y coordinates using least squares (without SciPy).
    Returns the circle's center (cx, cy) and radius (r).
    """
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Construct the system of equations
    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    b = x**2 + y**2

    # Solve for cx, cy, and c using least squares
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Compute the radius
    r = np.sqrt(cx**2 + cy**2 + c)

    return cx, cy, r

def find_best_frame(dir,tsi_files):
    """
    Takes a list of TSI file paths and finds the file with the most circular hole.
    Returns the file name with the lowest circularity deviation.
    """
    best_file = None
    best_circularity = float('inf')
    
    for file_path in tsi_files:
        tsi_instance = TSIFile(os.path.join(dir,file_path))
        broken_links = tsi_instance.validate_triangle_connectivity()
        
        if not broken_links:
            print(f"No boundary edges found in {file_path}, skipping.")
            continue
        
        edge_vertices = set()
        for edge in broken_links:
            edge_vertices.add(edge[0])
            edge_vertices.add(edge[1])
        
        x_coords, y_coords = [], []
        for vertex_idx in edge_vertices:
            vertex = tsi_instance.get_vertex_by_index(vertex_idx)
            x_coords.append(vertex['x'])
            y_coords.append(vertex['y'])
        
        x_coords, y_coords = np.array(x_coords), np.array(y_coords)
        cx, cy, r = fit_circle(x_coords, y_coords)
        deviations = np.abs(np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) - r)
        circularity_score = np.std(deviations)
        
        if circularity_score < best_circularity:
            best_circularity = circularity_score
            best_file = file_path
    
    return best_file

def natural_sort_key(s):
    """Extracts the numeric part from 'dtsXX.tsi' filenames for proper sorting."""
    parts = s.replace("dts", "").replace(".tsi", "")  # Remove 'dts' prefix and '.tsi' suffix
    return int(parts)  # Convert the remaining number to integer for correct sorting

def main():
    parser = argparse.ArgumentParser(description="modify equilibrations for productions")
    parser.add_argument('-txt', '--output', required=True, help="directories file")
    args = parser.parse_args()
    
    directories_file  = args.output

    with open(directories_file, 'r') as f:
        directories = [line.strip() for line in f if line.strip()]
    
    with open(f"results{directories_file.rsplit('.', 1)[0]}.txt", 'w') as f:
            f.write("R,Calc_R,lambda,tau,status\n")

    for i,directory in enumerate(directories):

        traj_file_path = os.path.join(directory,"TrajTSI")
        files = sorted(os.listdir(traj_file_path))
        print(len(files),"Found in trajectory files")

        files = sorted(os.listdir(traj_file_path), key=natural_sort_key)
        print(i)
        last_files = files[-100:]
        last_files = ["TrajTSI/"+last_file for last_file in last_files]
        best_file = find_best_frame(directory,last_files)
        print("and the best frame of the last 100 is:",best_file)

        rad_tsi = TSIFile(os.path.join(directory,best_file))
        ra = rad_tsi.calc_radius()

        # Extract parameters based on the structure
        parts = directory.split(os.sep)
        radius = float(parts[-4].split('_')[1])  # Extract radius from the 4th last part
        lambda_val = float(parts[-3].split('_')[1])  # Extract lambda from the 3rd last part
        tau = float(parts[-2].split('_')[1])  # Extract tau from the 2nd last part

        with open(f"results{directories_file.rsplit('.', 1)[0]}.txt", 'a') as f:
            f.write(f"{radius},{ra},{lambda_val},{tau}\n")

        plot_path = os.path.join(os.getcwd(),"plots_visuals")
        os.makedirs(plot_path, exist_ok=True)  
        rad_tsi.visualize_links(plot_path+ f"/links_vis_{ra:.2f}.png",ra)
        
        copy_path = os.path.join(directory,best_file)
        paste_path = os.path.join(directory,"prod","dts.tsi")

        os.system(f'cp "{copy_path}" "{paste_path}"')
        # rad_tsi.visualize_links()

if __name__ == "__main__":
    main()
