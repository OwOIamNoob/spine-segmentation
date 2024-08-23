from TPTBox import BIDS_FILE, NII
# import numpy as np 
from pathlib import Path
import scipy
import skimage

class Volume:
    def __init__(self,
               path):
        path = Path(path)
        parent = path.parent
        bids_file = TPTBox.BIDS_FILE(str(path), str(parent))
        self.image = bids_file.open_nii()
        # self.scale = np.array(self.image.GetSpacing())[::-1]
        # print(self.scale)
        # self.affine = self.image.GetDirection()
        self.array = self.image.get_array()
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
  
    def gaussian_kernel(radius=3, dim=3, detla=0.8):

        mean = radius // 2 + ((radius + 1) % 2) / 2 

        coef = np.identity(dim) * np.power(delta, 2)
        inv_coef = np.linalg.inv(coef)

        denominator = (2 * np.pi * (delta ** 2)) ** (dim / 2)

        grid = np.arange(radius)
        mesh = np.array(np.meshgrid(*[grid] * dim)) - mean

        mesh = np.apply_along_axis(lambda x: np.exp( - 0.5 * x.T @ inv_coef @ x ), 0, mesh) / denominator
        return mesh

    def construct_window(radius=3, mode='cube', blur=False, kernel_size=3, delta=0.8):
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
        if blur:
            kernel = Volume.gaussian_kernel(radius=kernel_size, dim=3, delta=delta)
            window = scipy.ndimage.convolve(window, kernel)
        return window

    def normalize(self):
      """ Recover 3d volumetric to standard axis spacing of 1
      """
      self.array = scipy.ndimage.zoom(self.array, self.scale, order=0)


    def convex_hull(self, label=0, radius=3, mode='cube', threshold=0.05, blur=False, kernel_size=3, delta=0.8):
        """ Return hull of a dense volumetric
            Can work on either binary or multiclass mask
        """
        # Radius to determine level of exclusion
        if label == 0:
            arr = self.array.copy()
        else:
            arr = (self.array == label).astype(float)

        convo_window = Volume.construct_window(radius=radius, 
                                                mode=mode, 
                                                blur=blur, 
                                                kernel_size=kernel_size, 
                                                delta=delta)
        mask = scipy.ndimage.convolve(arr, convo_window, mode='constant')
        return mask

    def reduce(self, radius=3, mode='cube', ):
        mask = np.zeros_like(self.array)
        for label in np.unique(self.array):
            if label == 0: 
                continue
            mask += self.convex_hull( label=label, 
                                    radius=radius, 
                                    mode=mode, 
                                    threshold=threshold, 
                                    blur=False, 
                                    kernel_size=3, 
                                    delta=0.8)
        # Make sure that 
        self.array = mask

path = "/work/hpc/spine-segmentation/data/dataset/spine_nii/masks/1_t1.nii.gz"
volume = Volume(path)
volume.reduce(radius=5, mode='cross', threshold=0.05, blur=True)
volume.image.set_array_(volume.array)
volume.image.save("/work/hpc/spine-segmentation/outputs/dummy/blurred_laplace.nii.gz")


