import streamlit as st
import requests
import tempfile
import os
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
from collections import defaultdict
import matplotlib.colors as mcolors
from PIL import Image
import io


# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="zFZeqWPuSUSI5eKaFXfr"
)


#A bunch of definitions for analysis

def detect_window(image_path, model_id="window-detector/2", confidence_threshold=0.25):
    """
    Processes a single image, extracts window data, and stores it in a list.

    Args:
        image_path: The path to the image to process.
        model_id: The Roboflow model ID. Defaults to "window-detector/2".

    Returns:
        A list of dictionaries, where each dictionary represents a detected window
        with its coordinates, dimensions, and confidence.
    """

    predictions = CLIENT.infer(image_path, model_id=model_id)
    window_data = []
    if predictions and predictions.get('predictions'):
        for prediction in predictions['predictions']:
            if prediction['class'] == 'window':  # Filter for windows only
                window_data.append({
                    'x_center': prediction['x'],
                    'y_center': prediction['y'],
                    'width': prediction['width'],
                    'height': prediction['height'],
                    'confidence': prediction['confidence'],
                    'image_path': image_path  # Add image path for reference
                })
    return window_data

def round_to_quarter(value):
    """Rounds a number to the nearest 0.25"""
    return round(value * 4) / 4

def cluster_coordinates(coords, eps=20):
    """Clusters coordinates to identify rows or columns."""
    clustering = DBSCAN(eps=eps, min_samples=2).fit(coords.reshape(-1, 1))
    unique_clusters = np.unique(clustering.labels_[clustering.labels_ >= 0])
    cluster_centers = []
    for cluster_id in unique_clusters:
        cluster_points = coords[clustering.labels_ == cluster_id]
        cluster_centers.append(np.mean(cluster_points))
    return sorted(cluster_centers)

def cluster_window_types(dimensions, eps=10):
    """Clusters windows based on dimensions to identify types."""
    clustering = DBSCAN(eps=eps, min_samples=2).fit(dimensions)
    window_types = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        if label >= 0:
            window_types[label].append(dimensions[i].tolist())

    type_dimensions = {}
    for type_id, dims in window_types.items():
        avg_dims = np.mean(dims, axis=0)
        type_dimensions[f'Type_{type_id}'] = {
            'width': avg_dims[0],
            'height': avg_dims[1],
            'count': len(dims)
        }
    return type_dimensions

def analyze_spacings(centers, pattern_data, eps=15, min_spacing=30):  # Reduced eps from 25 to 15
    """
    Analyzes spacing patterns using grid lines with more sensitive grouping
    """
    # Get grid positions
    col_positions = np.array(pattern_data['col_positions'])

    # Calculate distances between adjacent grid lines
    distances = []
    positions = []
    pairs = []  # Store the column indices for debugging

    # Calculate all adjacent grid spacings
    for i in range(len(col_positions) - 1):
        distance = col_positions[i+1] - col_positions[i]
        if distance > min_spacing:
            distances.append(distance)
            positions.append((col_positions[i] + col_positions[i+1]) / 2)
            pairs.append((i, i+1))  # Store column pair indices

    if not distances:
        return {}

    # Cluster the distances with stricter tolerance
    distances_array = np.array(distances).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=2).fit(distances_array)

    # For debugging
    # print("Distances:", distances)
    # print("Clustering labels:", clustering.labels_)

    spacing_patterns = {}
    for label in set(clustering.labels_):
        if label >= 0:  # Ignore noise points
            mask = clustering.labels_ == label
            cluster_distances = np.array(distances)[mask]
            cluster_positions = np.array(positions)[mask]
            cluster_pairs = [pairs[i] for i in range(len(pairs)) if mask[i]]

            mean_distance = np.mean(cluster_distances)
            std = np.std(cluster_distances)

            # Only group very similar distances
            if std < eps/2:  # More strict standard deviation threshold
                spacing_patterns[f'Spacing_{label}'] = {
                    'distance': mean_distance,
                    'count': len(cluster_distances),
                    'std': std,
                    'positions': cluster_positions.tolist(),
                    'pairs': cluster_pairs  # For debugging
                }

    return spacing_patterns

def standardize_window_types(window_data, pattern_data, pixels_per_foot):
    """Standardizes window dimensions within each type."""
    standard_dimensions = {}
    for type_id, type_info in pattern_data['window_types'].items():
        width_ft = round_to_quarter(type_info['width'] / pixels_per_foot)
        height_ft = round_to_quarter(type_info['height'] / pixels_per_foot)
        standard_dimensions[type_id] = {
            'width': width_ft * pixels_per_foot,
            'height': height_ft * pixels_per_foot
        }

    standardized_windows = []
    for window in window_data:
        for type_id, type_info in pattern_data['window_types'].items():
            if (abs(window['width'] - type_info['width']) < 10 and
                abs(window['height'] - type_info['height']) < 10):
                new_window = window.copy()
                new_window['width'] = standard_dimensions[type_id]['width']
                new_window['height'] = standard_dimensions[type_id]['height']
                standardized_windows.append(new_window)
                break

    # Update pattern_data with standardized dimensions
    for type_id, dims in standard_dimensions.items():
        pattern_data['window_types'][type_id].update(dims)

    return standardized_windows, pattern_data

def snap_to_grid(window_data, pattern_data):
    """Snaps window positions to the detected grid while preserving window types and dimensions."""
    row_positions = np.array(pattern_data['row_positions'])
    col_positions = np.array(pattern_data['col_positions'])

    standardized_windows = []
    for window in window_data:
        # Find nearest grid positions
        row_idx = np.argmin(np.abs(row_positions - window['y_center']))
        col_idx = np.argmin(np.abs(col_positions - window['x_center']))

        new_window = window.copy()
        new_window['x_center'] = col_positions[col_idx]
        new_window['y_center'] = row_positions[row_idx]

        standardized_windows.append(new_window)

    return standardized_windows

# Main Analysis Function
def analyze_facade(image_path, floor_height_ft=11.0):
    """Main analysis function"""
    # 1. Detect windows
    window_data = detect_window(image_path)
    if not window_data:
        print("No windows detected!")
        return None

    # 2. Extract centers and dimensions
    centers = np.array([[w['x_center'], w['y_center']] for w in window_data])
    dimensions = np.array([[w['width'], w['height']] for w in window_data])

    # 3. Basic pattern detection
    row_clusters = cluster_coordinates(centers[:, 1])
    col_clusters = cluster_coordinates(centers[:, 0])
    window_types = cluster_window_types(dimensions)

    # 4. Initialize pattern data
    pattern_data = {
        'num_rows': len(row_clusters),
        'num_cols': len(col_clusters),
        'row_positions': row_clusters,
        'col_positions': col_clusters,
        'window_types': window_types,
        'spacing': {
            'vertical': np.mean(np.diff(row_clusters))
        }
    }

    # 5. Calculate scaling
    vertical_spacing = np.mean(np.diff(row_clusters))
    pixels_per_foot = vertical_spacing / floor_height_ft

    # 6. Standardize dimensions and snap to grid
    window_data, pattern_data = standardize_window_types(window_data, pattern_data, pixels_per_foot)
    window_data = snap_to_grid(window_data, pattern_data)

    # 7. Update spacing patterns using grid lines
    centers = np.array([[w['x_center'], w['y_center']] for w in window_data])
    pattern_data['spacing']['patterns'] = analyze_spacings(centers, pattern_data)

    return window_data, pattern_data, pixels_per_foot


#Visualizations

def generate_analysis_text(pattern_data, pixels_per_foot):
    """Generates analysis information text with multiple spacing patterns."""
    info = [
        f"Grid: {pattern_data['num_rows']} rows × {pattern_data['num_cols']} columns",
        f"Vertical Spacing: {round_to_quarter(pattern_data['spacing']['vertical']/pixels_per_foot):.2f}ft",
        "\nHorizontal Spacing Patterns:"
    ]

    # Add spacing pattern information
    for pattern_id, pattern_info in pattern_data['spacing']['patterns'].items():
        spacing_ft = round_to_quarter(pattern_info['distance'] / pixels_per_foot)
        info.append(f"- {pattern_id}: {spacing_ft:.2f}ft ({pattern_info['count']} instances)")

    info.append("\nWindow Types:")
    for type_id, type_info in pattern_data['window_types'].items():
        width_ft = round_to_quarter(type_info['width'] / pixels_per_foot)
        height_ft = round_to_quarter(type_info['height'] / pixels_per_foot)
        info.append(f"- {type_id}: {width_ft:.2f}' × {height_ft:.2f}' ({type_info['count']} instances)")

    return "\n".join(info)

def visualize_facade_analysis(image_path, window_data, pattern_data, pixels_per_foot):
    """Visualizes the facade analysis with grid lines and typical spacing annotations."""
    fig, ax = plt.subplots(1, figsize=(15, 10))

    # Load and display faded image
    image = mpimg.imread(image_path)
    ax.imshow(image, alpha=0.3)

    # Generate colors for window types
    colors = list(mcolors.TABLEAU_COLORS.values())
    type_colors = {type_id: colors[i % len(colors)]
                  for i, type_id in enumerate(pattern_data['window_types'].keys())}

    # Draw grid lines
    for y in pattern_data['row_positions']:
        ax.axhline(y=y, color='blue', linestyle='--', alpha=0.4, linewidth=1.5)
    for x in pattern_data['col_positions']:
        ax.axvline(x=x, color='blue', linestyle='--', alpha=0.4, linewidth=1.5)

    # Draw windows
    for window in window_data:
        x, y = window['x_center'], window['y_center']
        w, h = window['width'], window['height']

        # Find window type
        window_type = None
        for type_id, type_info in pattern_data['window_types'].items():
            if (abs(w - type_info['width']) < 10 and
                abs(h - type_info['height']) < 10):
                window_type = type_id
                break

        if window_type:
            color = type_colors[window_type]

            # Draw fill
            rect_fill = patches.Rectangle(
                (x - w/2, y - h/2), w, h,
                linewidth=0,
                edgecolor='none',
                facecolor=color,
                alpha=0.1,
                zorder=2
            )
            ax.add_patch(rect_fill)

            # Draw edge
            rect_edge = patches.Rectangle(
                (x - w/2, y - h/2), w, h,
                linewidth=3,
                edgecolor=color,
                facecolor='none',
                alpha=1.0,
                zorder=3
            )
            ax.add_patch(rect_edge)

            # Add dimensions
            width_ft = round_to_quarter(w / pixels_per_foot)
            height_ft = round_to_quarter(h / pixels_per_foot)
            ax.text(x, y, f'{width_ft:.2f}\' x {height_ft:.2f}\'',
                   horizontalalignment='center',
                   verticalalignment='center',
                   color='black',
                   fontweight='bold',
                   bbox=dict(facecolor='white',
                            edgecolor=color,
                            alpha=0.8,
                            pad=2,
                            linewidth=2),
                   zorder=3)

    # Draw spacing annotations
    spacing_colors = ['red', 'green', 'purple', 'orange']
    y_offsets = [-30, -50, -70, -90]  # Different heights for different spacing types

    # Add spacing annotations
    for idx, (pattern_id, pattern) in enumerate(pattern_data['spacing']['patterns'].items()):
        spacing_ft = round_to_quarter(pattern['distance'] / pixels_per_foot)
        color = spacing_colors[idx % len(spacing_colors)]

        # Find one good position for this spacing pattern
        y_pos = pattern_data['row_positions'][0] + y_offsets[idx % len(y_offsets)]

        for pos in pattern['positions']:
            ax.annotate(
                f'{spacing_ft:.2f}\'',
                xy=(pos - pattern['distance']/2, y_pos),
                xytext=(pos + pattern['distance']/2, y_pos),
                color=color,
                weight='bold',
                arrowprops=dict(arrowstyle='<->',
                               color=color,
                               lw=2),
                horizontalalignment='center',
                verticalalignment='bottom',
                zorder=4
            )

    # Add analysis info
    info_text = generate_analysis_text(pattern_data, pixels_per_foot)
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round',
                     facecolor='white',
                     edgecolor='black',
                     alpha=0.9,
                     pad=1),
            zorder=5)

    # Add legend
    handles = [patches.Patch(color=color,
                           label=type_id,
                           alpha=0.2,
                           linewidth=2)
              for type_id, color in type_colors.items()]
    ax.legend(handles=handles,
             title="Window Types",
             bbox_to_anchor=(1.05, 1),
             loc='upper left',
             framealpha=0.9,
             edgecolor='black')

    plt.title("Facade Pattern Analysis", pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Main wrapper function
def analyze_facade_image(image_path, floor_height_ft=10.0):
    """Convenience function to run complete analysis on a facade image."""
    results = analyze_facade(image_path, floor_height_ft)
    if results:
        window_data, pattern_data, pixels_per_foot = results
        visualize_facade_analysis(image_path, window_data, pattern_data, pixels_per_foot)
        return window_data, pattern_data, pixels_per_foot
    return None



#Streamlit interface:
st.title("Window Grammar Analyzer")
st.write("Provide a URL to analyze window patterns.")

image_url = st.text_input("Enter image URL (jpg/png)")
floor_height = st.number_input("Floor Height (ft)", min_value=1, max_value=20, value=11)

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None

if st.button("Analyze Window Grammar"):
    if image_url:
        with st.spinner("Loading and analyzing image..."):
            temp_path = load_image_from_url(image_url)
            if temp_path:
                try:
                    # Display original image
                    st.image(image_url, caption="Input Image", use_container_width=True)
                    
                    # Run analysis
                    plt.clf()  # Clear any existing plots
                    results = analyze_facade_image(temp_path, floor_height)
                    
                    if results:
                        fig = plt.gcf()  # Get the current figure
                        st.pyplot(fig)    # Display it in Streamlit
                    
                    # Clean up
                    os.unlink(temp_path)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    else:
        st.error("Please provide an image URL")
