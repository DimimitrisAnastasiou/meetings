import os
import numpy as np
import argparse

class MembraneSystem:
    def __init__(self, file_path):
        """
        Initialize the MembraneSystem class by loading the .q file and extracting system properties.
        """
        self.file_path = file_path
        self.box_size = None
        self.vertex_count = None
        self.triangle_count = None
        self.vertices = []
        self.triangles = []

        # Load the file content
        self.load_file()

    def load_file(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

            # First line contains the box size
            box_size_line = lines[0].split()
            self.box_size = [float(value) for value in box_size_line]

            # Second line contains the number of vertices
            self.vertex_count = int(lines[1].strip())

            # The next lines contain the vertex information (assume vertex count lines)
            for i in range(2, 2 + self.vertex_count):
                parts = lines[i].split()
                if len(parts) >= 4:
                    vertex = {
                        'index': int(parts[0]),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'z': float(parts[3]),
                        'property': int(parts[4]) if len(parts) > 4 else None
                    }
                    self.vertices.append(vertex)

            # After vertices, the remaining lines contain the triangle information
            self.triangle_count = int(lines[2 + self.vertex_count].strip())  # Number of triangles

            for i in range(3 + self.vertex_count, 3 + self.vertex_count + self.triangle_count):
                parts = lines[i].split()
                triangle = {
                    'index': int(parts[0]),
                    'v1': int(parts[1]),
                    'v2': int(parts[2]),
                    'v3': int(parts[3]),
                    'property': int(parts[4]) if len(parts) > 4 else None
                }
                self.triangles.append(triangle)

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
    def get_triangle_by_index(self, index):
        for triangle in self.triangles:
            if triangle['index'] == index:
                return triangle
        return None

    def find_vertices_within_circle(self, radius):
        """
        Finds all vertices within a given radius from the center point in the XY plane.
        :param radius: The radius of the circle.
        :return: A list of IDs of the vertices within the circle.
        """
        center_x = self.box_size[0] / 2
        center_y = self.box_size[1] / 2
        vertex_ids_within_circle = []
        
        for vertex in self.vertices:
            # Calculate the distance from the vertex to the center point in the XY plane
            distance = np.sqrt((vertex['x'] - center_x) ** 2 + (vertex['y'] - center_y) ** 2)
            
            # Check if the vertex is within the specified radius
            if distance <= radius:
                vertex_ids_within_circle.append(vertex['index'])
        
        return vertex_ids_within_circle

    def remove_vertices_by_ids(self, vertex_ids_to_remove):
        """
        Removes vertices by their IDs and also removes triangles that reference these vertices.
        Reindexes the remaining vertices sequentially.
        :param vertex_ids_to_remove: A list of vertex IDs to be removed.
        """
        # Create a set for faster lookup
        vertex_ids_to_remove_set = set(vertex_ids_to_remove)

        # Filter out vertices to be removed and create a mapping
        remaining_vertices = [v for v in self.vertices if v['index'] not in vertex_ids_to_remove_set]
        old_to_new_index = {v['index']: new_index for new_index, v in enumerate(remaining_vertices)}

        # Update vertices with new indices
        for new_index, vertex in enumerate(remaining_vertices):
            vertex['index'] = new_index
        self.vertices = remaining_vertices
        self.vertex_count = len(self.vertices)

        # Remove triangles that reference any removed vertices
        remaining_triangles = [
            triangle for triangle in self.triangles 
            if triangle['v1'] not in vertex_ids_to_remove_set 
            and triangle['v2'] not in vertex_ids_to_remove_set 
            and triangle['v3'] not in vertex_ids_to_remove_set
        ]

        # Update vertex indices in remaining triangles to reflect new indices
        for triangle in remaining_triangles:
            triangle['v1'] = old_to_new_index[triangle['v1']]
            triangle['v2'] = old_to_new_index[triangle['v2']]
            triangle['v3'] = old_to_new_index[triangle['v3']]
                
        self.triangles = remaining_triangles
        self.triangle_count = len(self.triangles)

    def write_to_file(self, output_file_path):
        """
        Writes the current membrane system (vertices and triangles) to a .q file.
        :param output_file_path: The path to the output .q file.
        """
        with open(output_file_path, 'w') as file:
            # Box size
            file.write(f"{self.box_size[0]} {self.box_size[1]} {self.box_size[2]}\n")
            # Number of vertices
            file.write(f"{self.vertex_count}\n")
            # Vertex information
            for vertex in self.vertices:
                if vertex['property'] is not None:
                    file.write(f"{vertex['index']} {vertex['x']} {vertex['y']} {vertex['z']} {vertex['property']}\n")
                else:
                    file.write(f"{vertex['index']} {vertex['x']} {vertex['y']} {vertex['z']}\n")
            # Number of triangles
            file.write(f"{self.triangle_count}\n")
            # Triangle information
            for triangle in self.triangles:
                if triangle['property'] is not None:
                    file.write(f"{triangle['index']} {triangle['v1']} {triangle['v2']} {triangle['v3']} {triangle['property']}\n")
                else:
                    file.write(f"{triangle['index']} {triangle['v1']} {triangle['v2']} {triangle['v3']}\n")

    def validate_triangle_connectivity(self):
        """
        Validates the connectivity of triangles to ensure each edge has a corresponding mirror.
        """
        from collections import defaultdict

        # Dictionary to store edges and their corresponding triangle index
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

def main():
    parser = argparse.ArgumentParser(description="Generate simulation directories and files.")
    parser.add_argument('-q', '--inputq', required=True, help="Input .q file path")
    parser.add_argument('-o', '--output', default="simulations", help="Output base directory")
    args = parser.parse_args()
    
    input_q_file = args.inputq
    base_dir = args.output
    radius = 8.0

    # Create modified membrane system with a hole of the specified radius
    membrane_system = MembraneSystem(input_q_file)

    # Find vertices within the specified radius and remove them to create a hole
    vertex_ids_to_remove = membrane_system.find_vertices_within_circle(radius)
    membrane_system.remove_vertices_by_ids(vertex_ids_to_remove)

    # Write the modified .q file to the simulation directory
    q_output_file = os.path.join(os.getcwd(), "topol.q")
    membrane_system.write_to_file(q_output_file)

    top_file = os.path.join(os.getcwd(), "top.top")
    with open(top_file, "w") as f:
        f.write(f"topol.q 9\n")

if __name__ == "__main__":
    main()