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

