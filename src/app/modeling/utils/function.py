def get_var(component, dim=0):
  coef, offset = component.pca_surface(axis=dim)

  return np.sum(component.pcd * coef, axis=1) - offset

def plot_var(component):
  def gradient(input):
    kernel = np.Gauss
  plt.close()
  pca_var = [get_var(component, dim) for dim in range(3)]
  pca_plot = [np.histogram(var, 100) for var in pca_var]
  print(len(pca_plot))
  for i, plot in enumerate(pca_plot):
    print(plot[0].shape, plot[1].shape)
    plt.plot(plot[1][1:], plot[0], label="Axis {}".format(i + 1))
  plt.legend()
  plt.show()

def surface_distance(pcd, center, normal):
  bias = np.sum(center * normal)
  return np.sum(pcd * normal, axis = 1) - bias

def project(pcd, normal, origin, angle=0.):
  displacement = pcd - origin
  # print(displacement.shape)
  dist = np.sum(displacement @ normal[:, None], axis=1)[:, None] / np.linalg.norm(normal) * normal * (1 + np.tan(angle))
  print(dist.shape)
  return pcd - dist

def project2d(pcd, normal, origin):
  # TO decompose a 3d vector to sum of 2 pependicular basis vector, output a 2-d scatters
  displacement = pcd - origin
  # print(displacement.shape, normal[0].shape)
  d1 = (displacement @ normal[0])[:, None] / np.linalg.norm(normal[0])
  d2 = (displacement @ normal[1])[:, None] / np.linalg.norm(normal[1])
  return np.concatenate([d1, d2], axis=1)