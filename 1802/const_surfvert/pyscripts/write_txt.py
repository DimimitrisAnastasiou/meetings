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

def main():
    parser = argparse.ArgumentParser(description="modify equilibrations for productions")
    parser.add_argument('-txt', '--output', required=True, help="directories file")
    args = parser.parse_args()
    
    directories_file  = args.output

    output_folder = f"txts_from_{directories_file[:-4]}"

    with open(directories_file, 'r') as f:
        directories = [line.strip() for line in f if line.strip()]
    
    with open(os.path.join(f"info_from_{directories_file}.txt"), 'w') as f:
        f.write(f"Radius,Calc_Radius,Lambda,Tau,Status\n")

    for i,directory in enumerate(directories):

        radius_file_path = os.path.join(directory,"dts.tsi")
        radius_tsi = TSIFile(radius_file_path)

        print(i)
        result = process_tsi_files(directory)
        if result is None:
            continue

        first_avg, last_avg, status = result

        # Extract simulation parameters from directory structure (assumes */radius_{radius}/lambda_{lambda}/tau_{tau})
        
        try:
            parts = directory.split(os.sep)

            # Ensure the directory has the expected number of parts
            if len(parts) < 4:
                raise ValueError(f"Invalid directory structure: {directory}")

            # # Extract parameters based on the expected structure
            radius = float(parts[-5].split('_')[1])  # Extract radius from the 5th last part
            lambda_val = float(parts[-4].split('_')[1])  # Extract lambda from the 4th last part
            tau = float(parts[-3].split('_')[1])  # Extract tau from the 3rd last part

        except (IndexError, ValueError) as e:
            print(f"Error processing directory: {directory}. {e}")
            continue

        ra  = radius_tsi.calc_radius()

        with open(os.path.join(f"info_from_{directories_file}.txt"), 'a') as f:
            f.write(f"{radius:.2f},{ra:.2f},{lambda_val:.2f},{tau:.2f},{status}\n")

        write_simulation_info(output_folder, radius, lambda_val, tau, status)

if __name__ == "__main__":
    main()
