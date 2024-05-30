import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift,ifftshift,ifft2


def fourier_transform(image):
    if len(image.shape) == 3:
        image = np.mean(image, axis=-1)

    # Perform 2D Fourier Transform
    fft_result = fft2(image)

    # Shift zero frequency components to the center
    fft_result_shifted = fftshift(fft_result)

    # Calculate the magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = np.log(np.abs(fft_result_shifted) + 1)

    return magnitude_spectrum,fft_result_shifted

def inverse_fourier(fft_result_shifted):
    # Inverse shift to prepare for inverse Fourier transform
    fft_result_unshifted_masked = ifftshift(fft_result_shifted)
    # Inverse Fourier Transform
    original_filtered = np.abs(ifft2(fft_result_unshifted_masked))
    return original_filtered


def add_rotation(angle, image):
    # Define the rotation angle (in degrees)

    # Get the center of the image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def add_translation(shift_x, shift_y, image):
    # Define the rotation angle (in degrees)

    height, width = image.shape[:2]

    # Define the translation matrix
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Apply the translation to the rotated image
    shifted_image = cv2.warpAffine(image, translation_matrix, (width, height))

    return shifted_image
        

image = plt.imread('F:\FYP\FYP\MRART_png_scans\scan_000103\standard_000103_png.png')
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

magnitude_spectrum,fft_result_shifted = fourier_transform(image)

rotated_image = add_translation(1,0,image)
rotated_magnitude_spectrum,rotated_fft_result_shifted = fourier_transform(rotated_image)


for number_of_lines in range(10,100,1):
    width = 1
    line_indexes = np.random.randint(0, rotated_image.shape[1]-2, number_of_lines)
    print(number_of_lines)
    motion_corrupted = fft_result_shifted.copy()
    for i in line_indexes:
        extracted = rotated_fft_result_shifted[:,i:i+width]
        motion_corrupted[:,i:i+width] = extracted
        # extracted = rotated_fft_result_shifted[i:i+width,:]
        # motion_corrupted[i:i+width,:] = extracted

    inverse_fourier_corrupted = inverse_fourier(motion_corrupted)
    # plt.imshow(inverse_fourier_corrupted,cmap='gray')
    # plt.show()
    plt.imsave(f"F:\FYP\Different_noise_levels_4\{number_of_lines}.png",inverse_fourier_corrupted,cmap="gray")

plt.imsave(f"F:\FYP\Different_noise_levels_4\Original.png",image,cmap="gray")