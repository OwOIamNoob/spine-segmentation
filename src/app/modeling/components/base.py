import nibabel as nib
import numpy as np


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
    
  def normalize(self):
      """ Recover 3d volumetric to standard axis spacing of 1
      """
      self.array = scipy.ndimage.zoom(self.array, self.scale, order=0)

############## Base component for every part

class Component:
    def __init__(self, volume, index):
        self.volume = volume
        self.index = index
        self.center = None
        self.pcd = self.get_pcd()
        self.iso_surface = dict()


    def get_pcd(self):
        return np.stack(np.where(self.volume == self.index)).T

    def get_pcd_fig(self, mode='markers', marker=None):
        if self.pcd is None:
          self.pcd = self.get_pcd()

        return go.Scatter3d(
                x=self.pcd[:, 0], y=self.pcd[:, 1], z=self.pcd[:, 2],
                mode=mode,
                marker=dict(size=1, color=self.cmap) if not marker else marker,
                name=str(self.index)
                )


    def get_mesh(self, cache=False, plot=False, spacing=(1., 1., 1.)):
        verts, faces, normals, values = skimage.measure.marching_cubes(self.pcd, 
                                                                        0,
                                                                        spacing=spacing,
                                                                        gradient_direction='ascent',
                                                                        step_size=1,
                                                                        method='lorensen')
        # Save into component cache
        if cache is True: 
            # This equal to surfacing
            self.iso_surface['verts'] = verts

            # Mesh properties
            self.iso_surface['faces'] = faces
            self.iso_surface['normals'] = normals
            self.iso_surface['values'] = values

        return go.Mesh3d(
                x=verts[:, 0], y= verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                name=str(self.index)
                ) if plot is True else None
    
