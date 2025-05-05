class Vertebrae(PCAComponent):
  def __init__(self, volume, index, cmap, orient):
    super().__init__(volume, index, cmap, orient)
    assert index < 100, "Vert region is singular"
    super().estimate_pca('full')
    self.body = PCAComponent(volume.copy(), index, cmap, orient)
    self.ring = PCAComponent(volume.copy(), index, cmap, orient)

  def crop_body(self, upper_disk, lower_disk, resolution=[100, 100], proj_angle= 0.05 * np.pi, endplate_quantile=0.2, hist_degree=2, threshold = 0.6):
    assert upper_disk is not None or lower_disk is not None, "At least 1 disk is required"
    disk_pcd = []
    disk_axis = []
    if upper_disk is not None:
      upper_disk.crop_pca_(threshold=[None, None, [0.05, 0.95]])
      upper_disk.estimate_pca('full')
      upper, lower, _ = upper_disk.split_plate(axis=2, quantile=endplate_quantile)
      upper_disk.update_(pcd=lower)
      upper_disk.estimate_pca('full')
      disk_pcd.append(upper_disk.pcd)
      disk_axis.append(upper_disk.normals)

    if lower_disk is not None:
      lower_disk.crop_pca_(threshold=[None, None, [0.05, 0.95]])
      lower_disk.estimate_pca('full')
      upper, lower, _ = lower_disk.split_plate(axis=2, quantile=endplate_quantile)
      lower_disk.update_(pcd=upper)
      lower_disk.estimate_pca('full')
      disk_pcd.append(lower_disk.pcd)
      disk_axis.append(lower_disk.normals)

    # Add vert un-normalized
    disk_axis.append(self.normals)
    disk_pcd = np.concatenate(disk_pcd, axis=0)
    disk_axis = np.array(disk_axis)

    disk_center = np.mean(disk_pcd, axis=0)
    proj_axis = np.mean(disk_axis, axis=0)
    print(disk_axis.shape, proj_axis.shape)
    proj_origin = project(disk_center.copy()[None, :], proj_axis[2], self.center, 0.)
    vert_center_proj = project(self.center.copy()[None, :], proj_axis[2], proj_origin, proj_angle)
    vert_proj = project(self.pcd.copy(), proj_axis[2], proj_origin, proj_angle)

    vert_proj_2d = project2d(vert_proj, proj_axis[:2], proj_origin)
    vert_center_proj_2d = project2d(vert_center_proj, proj_axis[:2], proj_origin)

    # Extract endplate ROI
    vert_img, xedges, yedges = np.histogram2d(vert_proj_2d[:, 0], vert_proj_2d[:, 1], bins=resolution)
    vert_center_quantized = [np.digitize(vert_center_proj_2d[0, 0], xedges), np.digitize(vert_center_proj_2d[0, 1], yedges)]
    angle = np.arctan((vert_center_quantized[0] - resolution[0] / 2) / (vert_center_quantized[1] - resolution[1] / 2))
    rotation_matrix = cv2.getRotationMatrix2D((resolution[1] / 2, resolution[0] / 2), angle * 180 / np.pi, scale=1)

    # Upscale image and centering for lossless transformation
    corner = np.zeros((4, 3))
    corner[1:3, 1] += resolution[0]
    corner[2:4, 0] += resolution[1]
    corner[:, 2] = 1

    transformed_corner = np.vstack([np.matmul(rotation_matrix, c.T) for c in corner])
    center_translation = [- min(transformed_corner[:, 0]), - min(transformed_corner[:, 1])]

    w1 = max(transformed_corner[:, 0]) - min(transformed_corner[:, 0])
    h1 = max(transformed_corner[:, 1]) - min(transformed_corner[:, 1])
    rotation_matrix[:, 2] += center_translation

    # Inverse transformation
    inv_rotation_matrix = cv2.getRotationMatrix2D((w1 / 2, h1 / 2), -angle * 180 / np.pi, scale=1)
    inv_rotation_matrix[:, 2] -= center_translation

    vert_img = cv2.warpAffine(vert_img, rotation_matrix, (int(w1), int(h1)))

    spacing = np.linspace(0, 1, 256)
    hist_diag  = (spacing ** hist_degree) * vert_img.max()
    vert_img = np.interp(cv2.GaussianBlur(vert_img, (5, 7), sigmaX=0.5, sigmaY=1.), spacing * vert_img.max(), hist_diag)

    # Kernel for morphology operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    level = cv2.threshold(vert_img, np.quantile(vert_img[vert_img > 0], threshold), 255, cv2.THRESH_TOZERO | cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]

    # Get level histogram to eliminate conjunction
    values, freq = np.unique(level, return_counts=True)
    freq = np.cumsum(freq)
    freq = freq / freq[-1]

    level[level < np.quantile(level, np.quantile(freq, 0.5))] = 0
    level = cv2.morphologyEx(level, cv2.MORPH_CLOSE, kernel, iterations=2)
    print(level.shape, level.max(), level.min())

    # return vert_img, level, xedges, yedges

    # Extract connected component
    num_regs, labels, stats, centroid = cv2.connectedComponentsWithStats(level.astype(np.uint8), 4)
    body_index = np.argmax(stats[1:, 4]) + 1
    labels[labels != body_index] = 0
    labels[labels > 0] = 255
    labels = cv2.blur(labels, (3, 7))
    labels[labels > 0] = 255

    # Find convex hull of the component
    contour = cv2.findContours(labels.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contour[0][0])
    labels = cv2.fillPoly(labels, [hull], 255)
    labels = cv2.warpAffine(labels.astype(np.uint8), inv_rotation_matrix, (resolution[1], resolution[0]))

    vert_px_id = np.stack([np.digitize(vert_proj_2d[:, 0], xedges), np.digitize(vert_proj_2d[:, 1], yedges)], axis=1)
    vert_px_id[:, 0] = np.where(vert_px_id[:, 0] >= resolution[0], resolution[0] - 1, vert_px_id[:, 0])
    vert_px_id[:, 1] = np.where(vert_px_id[:, 1] >= resolution[1], resolution[1] - 1, vert_px_id[:, 1])

    # any particles standing in front of vert center is always accepted as body
    pos_pick = vert_proj_2d[:, 1] > vert_center_proj_2d[0, 1]
    picked = np.array([labels[id[0], id[1]] for id in vert_px_id])
    condition = (picked > 0) + pos_pick
    self.body.update_(pcd = vert.pcd[condition].copy())
    self.ring.update_(pcd = vert.pcd[~condition].copy())
    return self.body.pcd.copy(), self.ring.pcd.copy(), cv2.warpAffine(vert_img, inv_rotation_matrix, (resolution[1], resolution[0])), cv2.warpAffine(level, inv_rotation_matrix, (resolution[1], resolution[0])), labels

  def estimate_pca(self, mode = 'full'):
    super().estimate_pca(mode)
    self.body.estimate_pca(mode)
    self.ring.estimate_pca(mode)



