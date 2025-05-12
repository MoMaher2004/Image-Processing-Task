import cv2
import numpy as np

def apply_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def apply_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = image.copy()
    total_pixels = image.size
    
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255
    
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy

def apply_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape if len(image.shape) == 3 else (*image.shape, 1)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_low_pass_filter(image, size):
    kernel = np.ones((size, size), np.float32) / (size * size)
    return cv2.filter2D(image, -1, kernel)

def apply_high_pass_filter(image, size):
    kernel = -np.ones((size, size), np.float32)
    center = size // 2
    kernel[center, center] = size * size - 1
    return cv2.filter2D(image, -1, kernel)

def apply_mean_filter(image, size):
    return cv2.blur(image, (size, size))

def apply_average_filter(image, size):
    return cv2.boxFilter(image, -1, (size, size), normalize=True)

def apply_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))
    return laplacian

def apply_gaussian(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_vert_sobel(image):
    Sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Sobel = np.uint8(np.clip(Sobel, 0, 255))
    return Sobel

def apply_horiz_sobel(image):
    Sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    Sobel = np.uint8(np.clip(Sobel, 0, 255))
    return Sobel

def apply_vert_prewitt(image):
    kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def apply_horiz_prewitt(image):
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    return cv2.filter2D(image, -1, kernel)

def apply_line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    output = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return output

def apply_circles_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    output = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
    return output

def apply_dilation_filter(image, size):
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def apply_erosion_filter(image, size):
    kernel = np.ones((size, size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_close_filter(image, size):
    kernel = np.ones((size, size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_open_filter(image, size):
    kernel = np.ones((size, size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
