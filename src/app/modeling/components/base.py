import nibabel as nib
import numpy as np
import TPTBox as tpt
from pathlib import Path

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

class TPTVolume:
  def __init__(self,
               path):
    path = Path(path)
    parent = path.parent
    bids_file = tpt.BIDS_FILE(str(path), str(parent))
    self.image = bids_file.open_nii()
    self.image.seg = True
    # self.scale = np.array(self.image.GetSpacing())[::-1]
    # print(self.scale)
    # self.affine = self.image.GetDirection()
    self.array = self.image.get_array()
    self.spacing = self.image.zoom
    # print("Label ordering:", np.unique(self.array))

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

  def construct_window(radius=3, mode='cube'):
      """ Constructing window for non-equal supression
      """
      assert radius % 2 != 0
      r = radius // 2
      # add noise to prevent neighbor exclusion
      window = np.ones([radius, radius, radius]) + np.random.randn(radius, radius, radius) * 0.3
      x = np.linspace(-r, r, radius)
      y, z = x.copy(), x.copy()
      xv, yv, zv = np.meshgrid(x, y, z)

      if mode == 'cyclic':
        window[xv**2 + yv**2 + zv**2 >= (r + 0.25) ** 2] = 0
      elif mode == 'cross':
        window *= (xv == 0) | (yv == 0) | (zv == 0)
      elif mode == 'uni-cross':
        x = xv == 0
        y = yv == 0
        z = zv == 0
        window *= (x & y) | (y & z) | (z & x)
      window[r][r][r] = 0
      window[r][r][r] = -np.sum(window)

      return window

  def normalize(self):
      """ Recover 3d volumetric to standard axis spacing of 1
      """
      self.image.rescale_([1, 1, 1], verbose=True, mode='nearest', c_val=0.)
      self.array = self.image.get_array()

  def recover(self):
      self.image.set_array_(self.array)
      self.image.rescale_(self.spacing, verbose=True, mode='nearest', c_val=0.)
      self.array = self.image.get_array()

  def convex_hull(self, label=0, radius=3, mode='cube', threshold=0.05):
      """ Return hull of a dense volumetric
          Can work on either binary or multiclass mask
      """
      # Radius to determine level of exclusion
      if label == 0:
        arr = self.array.copy()
      else:
        arr = (self.array == label).astype(float)

      convo_window = TPTVolume.construct_window(radius=radius, mode=mode)
      mask = scipy.ndimage.convolve(arr, convo_window, mode='constant')
      optimal = abs(mask) >= threshold
      return optimal.astype(int)

  def reduce(self, radius=3, mode='cube', threshold=1):
      mask = np.zeros_like(self.array).astype(int)
      for label in np.unique(self.array):
        if label == 0:
          continue
        mask += self.convex_hull( label=label, radius=radius, mode=mode, threshold=threshold)
      mask = mask > 0
      # Make sure that
      self.array[~mask] = 0

  def update_(self, array: np.ndarray):
      self.image.set_array_(array)
      self.array = self.image.get_array()

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

class PCAComponent(Component):
  orient_map = {'L': False, 'R': True,
                'P': False, 'A': True,
                'S': False, 'I': True}

  def __init__(self, volume, index, cmap, orient):
    super().__init__(volume, index, cmap)
    self.normals = None
    self.variance = None
    self.center = None
    self.orient = [PCAComponent.orient_map[char] for char in orient]
    self.concave = [True, True, True]

  def estimate_pca(self, mode="full"):
    if self.pcd is None:
      self.pcd = self.get_pcd()

    self.center = np.mean(self.pcd, axis=0)

    pca = PCA(n_components=3, svd_solver=mode)
    pca.fit(self.pcd - self.center)
    normals = pca.components_
    variance = pca.explained_variance_
    index = np.argmax(np.abs(normals), axis=1)
    # print(index)
    if self.normals is None:
      self.normals = np.zeros_like(normals)
      self.variance = np.zeros_like(variance)
    for id, i in enumerate(index):
      # print(id, i)
      # print(self.orient[i], normals[id][i] > 0)
      if self.orient[i] != (normals[id][i] > 0):
        self.normals[i] = -normals[id]
        self.concave[i] = False
      else:
        self.normals[i] = normals[id]
      self.variance[i] = variance[id]
    # print(self.normals)


  def pca_surface(self, axis: int = 2):
    """ Return the surface that cross through middle of the disc
    """
    ### PCA axis comes in variance decreasing order.
    ### From the ellipsoid shape of the disc, the least axis is the z-axis of the Disc
    if self.normals is None:
      self.estimate_pca(mode="full")

    surface_normal = self.normals[axis].copy()
    # a(x-x0) + b(y- y0) + c(z - z0) + d = 0
    d = np.sum(surface_normal * self.center)
    return surface_normal, d

  def split_plate(self, quantile=0.4, axis=2):
    if self.pcd is None:
      self.pcd = self.get_pcd()
    # Score can be used to calculate height.
    if isinstance(quantile, float):
      quantile = [quantile, quantile]
    # Calculate distance
    coef, offset = self.pca_surface(axis=axis)
    score = np.sum(self.pcd * coef, axis=1) - offset
    # Split by PCA surface at a margin of quantiled distance
    lower = self.pcd[score > np.quantile(score[score > 0], quantile[0])]
    upper = self.pcd[score < np.quantile(score[score < 0], quantile[1])]

    return upper, lower, score

  def crop_pca(self, threshold=[None, None, None]):
    if self.pcd is None:
      self.pcd = self.get_pcd()
    criteria = [True, True, True]
    for dim, bound in enumerate(threshold):
      if bound is None:
        continue
      if isinstance(bound, float):
        bound = [bound, 1 - bound]
      coef, offset = self.pca_surface(axis=dim)

      score = np.sum(self.pcd * coef, axis=1) - offset
      if bound[1] > bound[0]:
        crit = (score > np.quantile(score, bound[0])) & (score < np.quantile(score, bound[1]))
      else:
        crit = (score > np.quantile(score, bound[0])) | (score < np.quantile(score, bound[1]))
      criteria[dim] = crit

    criteria = criteria[0] & criteria[1] & criteria[2]
    return self.pcd[criteria].copy()

  def put(self, array, coordinate, value):
    for coord in coordinate:
      array[coord[0], coord[1], coord[2]] = value

  def update_(self, array: np.ndarray | None = None, pcd: np.ndarray | None = None):
    if array is None and pcd is None:
      return

    if array is not None:
      self.volume = array
      self.pcd = self.get_pcd()
    elif pcd is not None:
      self.pcd = pcd
      output = np.zeros_like(self.volume)
      self.put(output, pcd, self.index)
      self.volume = output

  def crop_pca_(self, threshold=[None, None, None]):
    self.update_(pcd=self.crop_pca(threshold=threshold))







