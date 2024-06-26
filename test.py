import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

img = cv2.imread("pictures/bin1.png", cv2.IMREAD_GRAYSCALE)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(img, cmap=plt.cm.Greys_r)

print(int(math.sqrt(img.shape[0]**2 + img.shape[1]**2)))

theta = np.linspace(0.0, 180.0, max(img.shape), endpoint=False)
print(theta)
sinogram = radon(img, theta=theta)
dx, dy = 0.5 * 180.0 / max(img.shape), 0.5 / sinogram.shape[0]
print(dx, dy)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    aspect='auto',
)

fig.tight_layout()
plt.show()