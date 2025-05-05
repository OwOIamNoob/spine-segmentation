class CobbAngle:
  def __init__(self, discs, threshold=0):
    # Sort in id ~ sort in z-position
    self.discs = sorted(discs, key=discs.index)
    ids = [disc.index for disc in self.discs]
    self.discs = dict(zip(ids, [[disc] for disc in self.discs]))
    self.threshold = threshold
    self.set_up()
    print(ids)



  def unit_vector(vector, offset=0):
    norm = np.linalg.norm(vector)
    return vector / norm, offset / norm, norm

  def vector_angle(v1, v2):
    angle = np.arccos(np.dot(v1, v2)) / np.pi * 180
    return angle

  def set_up(self):
    """  This function estimate plane of every discs and store them in a dict for easier processing   """
    format = ["obj", "u_coef", "l_coef"]
    for id in self.discs.keys():
      disc = self.discs[id][0]
      upper, lower, _ = disc.split_plate(distance=self.threshold)

      u_coef, u_offset = CobbAngle.regression_plane(upper)
      l_coef, l_offset = CobbAngle.regression_plane(lower)

      u_coef_norm, u_offset, u_norm = CobbAngle.unit_vector(u_coef, offset=u_offset)
      l_coef_norm, l_offset, l_norm = CobbAngle.unit_vector(l_coef, offset=l_offset)

      # zip and turn to dictionary
      obj = dict(zip(format, [disc,
                                 [u_coef_norm, u_offset, u_norm],
                                 [l_coef_norm, l_offset, l_norm]]))
      self.discs[id] = obj

  def regression_plane(pcd):
    yz_d = np.concatenate([pcd[:, 1:].copy(), np.ones([pcd.shape[0], 1])], axis=1)
    x = pcd[:, 0].copy()
    b, c, offset = np.linalg.lstsq(yz_d, x, rcond=None)[0]
    a = -1
    coef = np.array([a, b, c])
    return -coef, -offset

  def cobb_angle(self, lower_id, upper_id):
    """ Work with point cloud only, not volumetric
    """
    # The order of plate chosen by vertebrea is reversed to that of disc.
    u_norm = self.discs[upper_id]["l_coef"][0]
    l_norm = self.discs[lower_id]["u_coef"][0]

    # Now, for the angle, the angle between plane is actually the angle between their normal vector, proven in highschool :D
    # Assuming that the coordinate system of our model follows Cartesian coordinate system w/o translation
    # (meaning that we normalized our coordinate system to match original Oxyz forward)
    coronal_angle = CobbAngle.vector_angle(u_norm[::2], l_norm[::2])
    sagittal_angle = CobbAngle.vector_angle(u_norm[1:], l_norm[1:])
    axial_angle = CobbAngle.vector_angle(u_norm[:2], l_norm[:2])
    # No top down angle since its
    return coronal_angle, sagittal_angle, axial_angle

  def cobb_table(self):
    # Expect Cobb table to be upper triangle matrix because its sense of order
    ids = list(self.discs.keys())
    ksize = len(ids)
    result = np.zeros([ksize, ksize, 3], dtype=float)
    for i in range(ksize):
      for j in range(i, ksize):
        angles = np.array([self.cobb_angle(ids[i], ids[j])])
        result[i, j] = angles

    # Users split it themselves
    return result

  def mesh(coef, center, ksize):
    """ Create mesh of plane with normal vector and its offset
        Used for debugging
    """
    if  isinstance(coef[0], np.ndarray):
      normal, offset, _ = coef
    else:
      normal = coef[:2]
      offset = coef[2]

    y = np.linspace(center[1] - ksize[0], center[1] + ksize[0], 20)
    z = np.linspace(center[2] - ksize[1], center[2] + ksize[1], 20)
    yv, zv = np.meshgrid(y, z)
    xv = -(yv * normal[1] + zv * normal[2] + offset) / normal[0]
    return xv, yv, zv

  def surface_plot(self, id, coef_type='u_coef', size = 20):
    """ Wrapper for plane mesh to plotly graph figure
    """
    center = self.discs[id]["obj"].center
    coef = self.discs[id][coef_type]
    color = self.discs[id]["obj"].cmap
    xv, yv, zv = CobbAngle.mesh(coef, center, size)
    return go.Surface(x=xv,
                      y=yv,
                      z=zv,
                      name=str(id) + " surface")
