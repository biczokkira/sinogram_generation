import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom


def theta(start, number):
    angle = start / number

    t = np.arange(0.0, 180.0, angle)

    return t


def radon_transformation(img, theta):
    u, s = img.shape

    center = s // 2
    sinogram = np.zeros((s, len(theta)))

    for i, angle in enumerate(np.deg2rad(theta)):
        rotation_matrix = cv2.getRotationMatrix2D((center, center), np.rad2deg(angle), 1.0)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (s, u))
        projection = np.sum(rotated_image, axis=0)
        sinogram[:, i] = projection

    sinogram = cv2.rotate(sinogram, cv2.ROTATE_180)
    return sinogram


img = cv2.imread("pictures/Fig6_SheppLoganPhantom.png", cv2.IMREAD_GRAYSCALE)

if img.shape[0] == img.shape[1]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1.set_title("Original")
    ax1.imshow(img, cmap=plt.cm.Greys_r)

    theta_array = theta(180.0, img.shape[0])

    sinogram = radon_transformation(img, theta_array)

    # Display sinogram
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
else :
    print("Width and height should be equal.")