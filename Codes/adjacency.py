from graph_tool.all import *
import numpy as np
import matplotlib.pyplot as plt

# Function to load data from CSV file and create a graph
def create_graph_from_csv(csv_file):
    # Create an empty directed graph
    g = Graph(directed=True)

    # Define property maps for edge weights and edge types
    weight = g.new_edge_property("int")
    edge_type = g.new_edge_property("string")

    # Dictionary to map original vertex indices to graph-tool vertex objects
    vertex_map = {}

    # Read data from file and add edges to the graph
    with open(csv_file, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header line
            source, target, count, etype = map(str.strip, line.split(','))

            # Add source and target vertices if not already present
            if int(source) not in vertex_map:
                v_source = g.add_vertex()
                vertex_map[int(source)] = v_source
            else:
                v_source = vertex_map[int(source)]

            if int(target) not in vertex_map:
                v_target = g.add_vertex()
                vertex_map[int(target)] = v_target
            else:
                v_target = vertex_map[int(target)]

            # Add an edge between source and target vertices
            e = g.add_edge(v_source, v_target)

            # Set edge properties
            weight[e] = int(count)
            edge_type[e] = etype

    # Return the graph along with edge properties
    return g, weight, edge_type

csv_file_path = 'edges.csv'

# Create the graph and retrieve edge properties
graph, edge_weights, edge_types = create_graph_from_csv(csv_file_path)

# Print basic information of the graph
print("Number of nodes:", graph.num_vertices())
print("Number of edges:", graph.num_edges())

# Draw the graph and save the result
output_file = 'graph_output.png'
graph_draw(graph, output_size=(800, 800), output=output_file)
print(f"Graph visualization saved to {output_file}")

# Define a list of colors, one for each label
colors = ['sandybrown', 'green', 'blue', 'red']

# Create subplots with increased width and height space
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Adjust the layout manually by adjusting spacing between subplots
fig.subplots_adjust(wspace=0.5, hspace=0.5)

# Loop to plot each adjacency matrix in a subplot
for i, label in enumerate(['aa', 'dd', 'ad', 'da']):
    subgraph_csv_file = f'{label}.csv'

    # Create the graph and retrieve edge properties for the subgraph
    subgraph, subgraph_weights, subgraph_types = create_graph_from_csv(subgraph_csv_file)

    # Get the adjacency matrix
    adjacency_matrix = adjacency(subgraph)[:1500, :1500]

    # Choose the subplot based on the current iteration
    ax = axs[i // 2, i % 2]

    # Plot the adjacency matrix with a specific color for each matrix
    ax.spy(adjacency_matrix, markersize=0.7, aspect='auto', color=colors[i])
    ax.set_title(f'Adjacency Matrix for {label}')

    # Remove the numbers on the axes
    ax.set_xticks([])
    ax.set_yticks([])

# Save the plot as an image
plt.tight_layout()
plt.savefig('combined_adjacency_matrices.png')
plt.show()

hemisphere_csv_file_path = 'nodes.csv'

# Create an empty dictionary to store hemisphere information
hemisphere_dict = {}

with open(hemisphere_csv_file_path, 'r') as hem_file:
    hem_lines = hem_file.readlines()
    for hem_line in hem_lines[1:]:
        values = map(str.strip, hem_line.split(','))
        try:
            index, _, hemisphere, *_ = map(str.strip, hem_line.split(','))

            if hemisphere.lower() in ('left', 'right'):
                hemisphere_dict[int(index)] = hemisphere.lower()
        except StopIteration:
            pass

# Create a property map for vertex colors
vertex_colors = graph.new_vertex_property("vector<double>")

# Assign colors based on hemisphere information
for vertex in graph.vertices():
    index = int(vertex)
    hemisphere = hemisphere_dict.get(index, 'N/A')
    if hemisphere == 'left':
        vertex_colors[vertex] = [1, 0, 0, 1]  # Red color for left hemisphere
    elif hemisphere == 'right':
        vertex_colors[vertex] = [0, 0, 1, 1]  # Blue color for right hemisphere
    else:
        vertex_colors[vertex] = [0.5, 0.5, 0.5, 1]  # Gray color for unknown hemisphere

# Draw the graph with specified vertex colors using graph_tool
output_file_graph_tool = 'graph_with_colors.png'
graph_draw(graph, vertex_color=vertex_colors, output_size=(800, 800), output=output_file_graph_tool)

print(f"Graph visualization with colors saved to {output_file_graph_tool}")