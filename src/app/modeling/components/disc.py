class Disc(Component):
  def __init__(self, volume, index):
    super().__init__(volume, index)
    # component axis
    self.normals = None
    # Variance for each axis
    self.variance = None

    self.center = None
    self.upper = None
    self.lower = None
    self.score = None
    self.distance = 0

  def estimate_pca(self, mode="full"):
    """ Wrapper for scikit PCA API
    """
    if self.iso_surface is None:
      raise AssertionError("Missing surface data")
    
    self.center = np.mean(self.iso_surface['verts'], axis=0)
    pca = PCA(n_components=3, svd_solver=mode)
    pca.fit(self.iso_surface['verts'])
    
    # Get params
    self.normals = pca.components_
    self.variance = pca.explained_variance_


  def pca_surface(self):
    """ Return the surface that cross through middle of the disc
    """
    ### PCA axis comes in variance decreasing order.
    ### From the ellipsoid shape of the disc, the least axis is the z-axis of the Disc
    if self.normals is None:
      self.estimate_pca(mode="full")

    self.surface_normal = self.normals[2].copy()
    # a(x-x0) + b(y- y0) + c(z - z0) + d = 0
    d = np.sum(self.surface_normal * self.center)
    return self.surface_normal, d

  def split_plate(self, distance=0):
    """ Seperate the disc to 2 non-overlapping parts that 
    """
    if self.pcd is None:
      self.pcd = self.get_pcd()
    # Score can be used to calculate height.
    if self.upper is None or self.lower is None or self.distance != distance:
      coef, offset = self.pca_surface()
      self.score = np.sum(self.pcd * coef, axis=1) - offset
      self.lower = self.pcd[self.score > distance]
      self.upper = self.pcd[self.score < -distance]
    return self.upper, self.lower, self.score

