import nibabel as nib
import numpy as np
import cv2
import SimpleITK as sitk
import scipy
import skimage
from sklearn.decomposition import PCA
import open3d as o3d
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import zoom
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
import os

class Volume:
  def __init__(self,
               path,
               reader):
    reader.SetFileName(path)
    reader.ReadImageInformation()
    self.image = reader.Execute()
    self.scale = np.array(self.image.GetSpacing())[::-1]
    print(self.scale)
    self.affine = self.image.GetDirection()
    self.array = sitk.GetArrayFromImage(self.image)
    self.array = scipy.ndimage.zoom(self.array, self.scale, order=0)
    self.name = path.split('/')[-1].split('.')[0]
    print("Label ordering:", np.unique(self.array))

  def get_pcd(self, index=None, return_value=False):
    if index is None:
      pcd = np.stack(np.where(self.array > 0)).T
    else:
      pcd = np.stack(np.where(self.array == index)).T
    values = None
    if return_value is True:
      values = []
      for point in pcd:
        values.append(self.array[point])
      # N-3, N-1
    return pcd, values

  def eliminate_noise(self, pcd, exps=5, min_samples=50):
    clustering = DBSCAN(eps=5, min_samples=min_samples).fit(pcd)
    pcd = pcd[clustering.labels_ != -1]
    pcd = pcd[:, ::-1]
    
    return pcd
  
  def merge_image(self, vertebral_img, disk_img = 0, canal_img = 0):
        if isinstance(canal_img, int) == False:
            canal_img = np.stack([canal_img, np.zeros(canal_img.shape), canal_img], axis = -1)
            canal_img = canal_img / canal_img.max() / 2

        if isinstance(disk_img, int) == False:
            disk_img = np.stack([disk_img, disk_img, np.zeros(disk_img.shape)], axis = -1)
            disk_img = disk_img / disk_img.max()

        norm_vertebral = Normalize(vmin=vertebral_img.min(), vmax=vertebral_img.max())

        vertebral_img = cm.viridis(norm_vertebral(vertebral_img.cpu()))[:, :, :3] + disk_img + canal_img

        return vertebral_img

  def save_2D_image(self, predict_seg, save_path = None):
      # coronal_view = zoom(np.sum(predict_seg, axis=2).transpose(0, 2, 1), (1, 1, (predict_seg.shape[2]/predict_seg.shape[1]/6)) ## We need to zoom because the space of images is not the same
      coronal_view = torch.sum(predict_seg, dim=2)
      coronal_view = torch.rot90(coronal_view, 1, dims=(1, 2))

      sagittal_view = torch.sum(predict_seg, dim=1)
      sagittal_view = torch.rot90(sagittal_view, 1, dims=(1, 2))
      sagittal_view = sagittal_view[:,:,80:-50]

      ## coronal_view[0] is the vertebral, coronal_view[1] is the canal, coronal_view[2] is the disk
      ## sagittal_view[0] is the vertebral, sagittal_view[1] is the canal, sagittal_view[2] is the disk

      image1 = self.merge_image(coronal_view[0], coronal_view[2])
      image2 = self.merge_image(sagittal_view[0], sagittal_view[2], sagittal_view[1])
      coronal_view_0 = self.merge_image(coronal_view[0])
      coronal_view_2 = self.merge_image(coronal_view[2])
      sagittal_view_0 = self.merge_image(sagittal_view[0])
      sagittal_view_2 = self.merge_image(sagittal_view[2])

      # image2 = np.rot90(image2, 1)

      print(image1.shape)
      print(image2.shape)

      image_all = np.concatenate((coronal_view_0, coronal_view_2, image1, sagittal_view_0, sagittal_view_2, image2), axis = 1)
      image_all = (image_all * 255).astype(np.uint8)

    #   plt.imshow(image_all)
    #   plt.show()
      
      image_all = Image.fromarray(image_all)
      
      if save_path == None:
        save_path = "/work/hpc/spine-segmentation/notebooks/images"
        
      os.makedirs(save_path, exist_ok=True)
      
      save_path = os.path.join(save_path, self.name + ".png")
      image_all.save(save_path)
      
      return image_all
  
# using class for easiear management

#cmaps = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
class Disc:
  def __init__(self, pcd, name, cmap="#636EFA"):
    self.pcd = pcd
    self.name = name
    self.cmap = cmap
    self.normals = None
    self.variance = None
    self.center = None
    self.upper = None
    self.lower = None
    self.score = None
    self.mesh = None
    
    
  def mass_center(self):
    return np.mean(self.pcd, axis=0)

  def volume(self, mask):
    return np.sum(mask == self.index)
  
  def estimate_pca(self, pca_mode='auto'):
    pca = PCA(n_components=3, svd_solver=pca_mode)
    pca.fit(self.pcd)
    self.normals = pca.components_
    self.variance = pca.explained_variance_
    self.center = pca.mean_
    del pca
    
  def pca_surface(self, size = 20):
    if self.normals is None:
      self.estimate_pca(pca_mode="full")
      
    normal_vector = self.normals[2]
    center = self.center
    
    x = np.linspace(center[0] - size, center[0] + size, 20)
    y = np.linspace(center[1] - size, center[1] + size, 20)
    xv, yv = np.meshgrid(x, y)

    zv = -(normal_vector[0] * (xv - center[0]) + normal_vector[1] * (yv - center[1]) - normal_vector[2] * center[2])/normal_vector[2]
    
    self.mesh = [xv, yv, zv]
    
    return normal_vector,  self.center
  
  def split_plate(self, distance=1):
    if self.pcd is None:
      self.pcd = self.get_pcd()
      
    # Score can be used to calculate height.
    if self.upper is None or self.lower is None or self.distance != distance:
      normal_vector, center = self.pca_surface()
      
      self.score = np.sum((self.pcd - center) * normal_vector, axis = 1)
      # self.score = np.sum(self.pcd * coef, axis=1) - offset
      
      self.lower = self.pcd[self.score > distance]
      self.upper = self.pcd[self.score < -distance]
      
    return self.upper, self.lower, self.score
  
  def get_pcd_fig(self):
    return go.Scatter3d(
            x=self.pcd[:, 0], y=self.pcd[:, 1], z=self.pcd[:, 2],
            mode='markers',
            marker=dict(size=1, color=self.cmap),
        )
    
  def get_surface_plot(self):
    if self.mesh == None:
      self.pca_surface()
    
    return go.Surface(x=self.mesh[0],
                      y=self.mesh[1],
                      z=self.mesh[2],
                      name=str(self.name) + " surface") 
  
  
class CobbAngle:
    def __init__(self, disks_pcd, threshold=0):
        self.disks = None
        self.threshold = threshold
        self.setup_cluster(disks_pcd)
    
    def setup_cluster(self, disks_pcd):
        clustering = DBSCAN(eps=5, min_samples=150).fit(disks_pcd)
        
        # Loại bỏ nhiễu (-1 là nhiễu)
        filtered_pcd = disks_pcd[clustering.labels_ != -1]
        filtered_labels = clustering.labels_[clustering.labels_ != -1]
        
        cmaps = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#FFDD44']        
        # Divide into each cluster
        unique_labels = np.unique(filtered_labels)  # Unique cluster
        self.disks = [Disc(filtered_pcd[filtered_labels == label], label, cmaps[label]) for label in unique_labels]
        
        disks = []
        for disk in self.disks:
            upper_plane, lower_plane, _ = disk.split_plate(self.threshold)
            disks.append({'disk': disk, 'upper': Disc(upper_plane, 0), 'lower': Disc(lower_plane, 0)})
        
        self.disks = disks

    @staticmethod
    def unit_vector(vector, offset=0):
        norm = np.linalg.norm(vector)
        return vector / norm, offset / norm, norm

    @staticmethod
    def vector_angle(v1, v2):
        v1 = np.array(v1, dtype=np.float64)
        v2 = np.array(v2, dtype=np.float64)
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            raise ValueError("One of the vectors is a zero vector, cannot compute angle.")
        
        dot_product = np.dot(v1, v2) / (norm_v1 * norm_v2)
        dot_product = np.clip(dot_product, -1.0, 1.0) 

        angle = np.arccos(dot_product) * 180 / np.pi
        return angle

    
    @staticmethod
    def projection_vector(u, v):
        u = np.array(u)
        if v is None:
            return u
        v = np.array(v)
        
        return u - np.dot(u, v) * CobbAngle.unit_vector(v)[0] / np.linalg.norm(v)
    
    def cobb_table(self, normal_vector = None):
        """calculate cobb angle

        Args:
            normal_vector (_type_): normal vector of projection plane

        Returns:
            _type_: _description_
        """
        results = []
        for i in range(len(self.disks)):
            row = []
            for j in range (len(self.disks)):
                u = CobbAngle.projection_vector(self.disks[i]['upper'].pca_surface()[0], normal_vector)
                v = CobbAngle.projection_vector(self.disks[j]['lower'].pca_surface()[0], normal_vector)
                # angle = CobbAngle.vector_angle(self.disks[i]['upper'].pca_surface()[0], self.disks[j]['lower'].pca_surface()[0])
                # print(self.disks[i]['upper'].pca_surface()[0], self.disks[j]['lower'].pca_surface()[0])
                angle = CobbAngle.vector_angle(u, v)
                if j <= i:
                    angle = -1
                row.append(angle)
            results.append(row)
        
        return results


import gradio as gr
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import scipy
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# Import your existing Volume, Disc, and CobbAngle classes here
# ... (keep your existing classes as they are)

def process_mri(file_obj, view_mode=True):
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_obj.name)
    
    volume = Volume(file_obj.name, reader)
    
    # Define structures with their indices and softer colors
    structures = {
        "vertebra": {"index": 1, "color": "#FFB6C1"},  # Light pink
        "canal": {"index": 2, "color": "#98FB98"},     # Pale green
        "disk": {"index": 4, "color": "#87CEEB"},       # Sky blue
    }
    
    # Alternative color schemes you could try (just uncomment the one you prefer):
    
    # Pastel palette
    # structures = {
    #     "vertebra": {"index": 1, "color": "#FFD1DC"},  # Pastel pink
    #     "canal": {"index": 2, "color": "#B4EEB4"},     # Pastel green
    #     "disk": {"index": 4, "color": "#ADD8E6"}       # Pastel blue
    # }
    
    # Muted palette
    # structures = {
    #     "vertebra": {"index": 1, "color": "#E6B0AA"},  # Muted rose
    #     "canal": {"index": 2, "color": "#A2D9CE"},     # Muted teal
    #     "disk": {"index": 4, "color": "#AED6F1"}       # Muted blue
    # }
    
    # Create figure with white background
    fig = go.Figure(layout=dict(
        scene=dict(
            xaxis=dict(visible=True),
            yaxis=dict(visible=True),
            zaxis=dict(visible=True),
            aspectmode='data',
            bgcolor='white'  # Set background color to white
        ),
        width=800,
        height=600,
        paper_bgcolor='white'  # Set paper background to white
    ))
    
    threshold = {
        "vertebra": 250, 
        "canal": 350, 
        "disk": 150
    }
    
    if view_mode == "all":
        structures_to_show = structures
    else:
        # Show only the selected structure
        structures_to_show = {view_mode: structures[view_mode]}
    
    all_pcds = {}
    masks = []
    # Initialize a list to store masks instead of a dictionary
    
    # Process each structure
    for struct_name, struct_info in structures.items():
        pcd, _ = volume.get_pcd(index=struct_info["index"], return_value=False)
        pcd = volume.eliminate_noise(pcd, 5, threshold[struct_name])
        all_pcds[struct_name] = pcd
        
        # Create a mask directly and append to the list
        mask = np.zeros(volume.array.shape[::-1], dtype=int)
        mask[tuple(pcd.T)] = 1
        masks.append(mask)  # Store mask in a list instead of a dictionary
        del mask

        
        if struct_name in structures_to_show:
            # Add scatter plot with slightly larger points and opacity
            fig.add_trace(
                go.Scatter3d(
                    x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2],
                    mode='markers',
                    marker=dict(
                        size=0.8,  # Slightly larger points
                        color=struct_info["color"],
                        opacity=0.7  # Add some transparency
                    ),
                    name=struct_name
                )
            )
    
    # Update layout for better visibility
    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.9,
            xanchor="right",
            x=0.9
        )
    )
    
    image_all = volume.save_2D_image(torch.Tensor(np.array(masks)))
    # image_all = torch.Tensor(np.array(masks))
    # print(image_all)
    # image_all = None
    del masks
    return fig, all_pcds, image_all

def visualize_pca(pcd):
    if pcd is None:
        return None
    pcd = pcd['disk']
    
    cobb = CobbAngle(disks_pcd=pcd, threshold=1)
    
    disks_figs = [disk['disk'].get_pcd_fig() for disk in cobb.disks]
    upper_plane = [disk['upper'].get_surface_plot() for disk in cobb.disks]
    lower_plane = [disk['lower'].get_surface_plot() for disk in cobb.disks]
    
    fig = go.Figure(
        data=disks_figs + upper_plane + lower_plane,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                aspectmode='data',
            ),
            title=dict(text="Disk Analysis with PCA"),
            width=800,
            height=600
        )
    )
    
    return fig, cobb

def calculate_cobb_angles(cobb_obj):
    if cobb_obj is None:
        return "Please visualize PCA first"
    
    cobb_table = cobb_obj.cobb_table(np.array([0., 1., 0.], dtype=np.float64))
    df = pd.DataFrame(cobb_table, 
                     columns=[f"Disk {i+1}" for i in range(len(cobb_table[0]))],
                     index=[f"Disk {i+1}" for i in range(len(cobb_table))])
    df = df.round(2)
    df = df.replace(-1, "")
    
    # Find the maximum angle (excluding empty cells)
    numeric_df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, empty strings become NaN
    max_angle = numeric_df.max().max()
    
    def style_cells(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for i in range(len(df)):
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if val == "":
                    styles.iloc[i, j] = ''
                else:
                    try:
                        float_val = float(val)
                        if float_val == max_angle:
                            styles.iloc[i, j] = 'background-color: #FFD700; color: black; font-weight: bold'
                        elif float_val >= 0:
                            styles.iloc[i, j] = 'background-color: #F0F8FF; color: black; font-weight: bold'
                    except (ValueError, TypeError):
                        styles.iloc[i, j] = ''
        return styles

    styled_df = df.style.apply(style_cells, axis=None)\
                   .set_table_styles([
                       {'selector': 'th',
                        'props': [('background-color', '#4CAF50'),
                                  ('color', 'white'),
                                  ('font-weight', 'bold'),
                                  ('padding', '8px'),
                                  ('border', '1px solid #ddd')]},
                       {'selector': 'td',
                        'props': [('padding', '8px'),
                                  ('border', '1px solid #ddd'),
                                  ('text-align', 'center')]},
                       {'selector': 'caption',
                        'props': [('caption-side', 'bottom'),
                                  ('font-size', '14px'),
                                  ('padding', '8px')]},
                       {'selector': 'table',
                        'props': [('border-collapse', 'collapse'),
                                  ('width', '100%'),
                                  ('max-width', '100%'),  # Ensure table fits within container
                                  ('font-size', '14px'),
                                  ('margin', '25px 0'),
                                  ('box-shadow', '0 0 20px rgba(0,0,0,0.1)'),
                                  ('overflow-x', 'auto')]},  # Enable horizontal scrolling
                   ])\
                   .set_caption(f'<span style="color: black;">Maximum Cobb Angle: {max_angle:.2f}°</span>')
    
    html_output = f"""
    <div style='padding: 20px; border-radius: 10px; background-color: white; overflow-x: auto;'>
        <h3 style='color: #333; margin-bottom: 15px;'>Cobb Angle Analysis</h3>
        <div style='max-width: 100%; overflow-x: auto;'>
            {styled_df.to_html(table_id='cobb-angle-table', classes='cobb-table')}
        </div>
        <p style='color: #666; font-size: 12px; margin-top: 10px;'>
            * Empty cells indicate invalid angle measurements<br>
            Maximum Cobb Angle: {max_angle:.2f}°
        </p>
        <style>
            .cobb-table {{
                border-collapse: collapse;
                width: 100%;
                min-width: 600px;
                margin: 25px 0;
                font-size: 14px;
                text-align: center;
            }}
            .cobb-table th, .cobb-table td {{
                padding: 8px;
                border: 1px solid #ddd;
                white-space: nowrap;
            }}
            .cobb-table th {{
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }}
            #cobb-angle-table {{
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
        </style>
    </div>
    """
    
    return html_output

if __name__ == "__main__":
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Spine Analysis Tool")
        
        with gr.Row():
            # First column
            with gr.Column(scale=23):
                file_input = gr.File(label="Upload MRI File (.nii or .nii.gz)")
                view_mode = gr.Radio(
                    choices=["all", "disk", "vertebra", "canal"],
                    value="all",
                    label="View Mode",
                    info="Select structure(s) to display"
                )
                visualize_btn = gr.Button("Visualize MRI")
                pca_btn = gr.Button("Apply PCA Analysis")
                cobb_btn = gr.Button("Calculate Cobb Angles")
                
                image_output = gr.Image(label="2D Visualization")  # Add this line
                cobb_table = gr.HTML(label="Cobb Angles Table")  # Moved to first column

            
            # Second column
            with gr.Column(scale=20):
                plot_output = gr.Plot(label="3D Visualization")
                pca_plot = gr.Plot(label="PCA Analysis")
    
        # Store state (keep these outside the columns)
        pcd_state = gr.State()
        cobb_state = gr.State()
        
        # Store state
        pcd_state = gr.State()
        cobb_state = gr.State()
        
        # Update click handlers
        visualize_btn.click(
            process_mri,
            inputs=[file_input, view_mode],  # Added view_mode input
            outputs=[plot_output, pcd_state, image_output]  # Add image_output
        )
        
        pca_btn.click(
            visualize_pca,
            inputs=[pcd_state],
            outputs=[pca_plot, cobb_state]
        )
        
        cobb_btn.click(
            calculate_cobb_angles,
            inputs=[cobb_state],
            outputs=[cobb_table]
        )

    demo.launch(
        share=True,
        server_name="0.0.0.0",  # Allows external connections
        server_port=6870,       # Specify port number
    )
    