import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_sinogram(image, num_angles):
    height, width = image.shape
    middle = height // 2

    a = num_angles / height

    angles = np.arange(0.0, 180.0, a)
    radians = np.deg2rad(-angles)
    cos_vals = np.cos(radians)
    sin_vals = np.sin(radians)

    sinogram = np.zeros((height, len(angles)))

    for angle_idx, angle in enumerate(angles):
        for x in range(width):
            for y in range(height):
                diag = int(x * cos_vals[angle_idx] + y * sin_vals[angle_idx] - middle * (cos_vals[angle_idx] + sin_vals[angle_idx] - 1))
                if 0 <= diag < height:
                    sinogram[diag, angle_idx] += image[y, x]

    sinogram /= np.max(sinogram)
    return sinogram


def main():
    img = cv2.imread("pictures/Fig6_SheppLoganPhantom.png", cv2.IMREAD_GRAYSCALE)

    num_angles = 180
    sinogram = generate_sinogram(img, num_angles)

    dx, dy = 0.5 * 180.0 / max(img.shape), 0.5 / sinogram.shape[0]


    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Eredeti kép')
    plt.axis('off')

    plt.subplot(1, 2, 2)

    plt.imshow(sinogram,
               cmap=plt.cm.Greys_r,
               extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
               aspect='auto',)
    plt.title('Szinogram')
    plt.xlabel('Szinusz szög')
    plt.ylabel('Projektált távolság')
    plt.show()


if __name__ == "__main__":
    main()
