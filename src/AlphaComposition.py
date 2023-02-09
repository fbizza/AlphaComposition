import cv2
import numpy as np

def alpha_compose(img, num_squares, alpha):
    height, width, channels = img.shape
    img_copy = img.copy()
    for i in range(num_squares):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(x1, width)
        y2 = np.random.randint(y1, height)
        square = np.ones((y2-y1, x2-x1, channels), dtype=np.float32) * 0
        img_copy[y1:y2, x1:x2] = square * alpha + img_copy[y1:y2, x1:x2] * (1 - alpha)
    return img_copy

if __name__ == '__main__':
    image_path = "../MyImage.jpg"
    img = cv2.imread(image_path)
    num_squares = 5
    alpha = 0.7
    augmented_img = alpha_compose(img, num_squares, alpha)
    window_name = "Original Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.resizeWindow(window_name, img.shape[1]//2, img.shape[0]//2)
    window_name = "Alpha Composed Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, augmented_img)
    cv2.resizeWindow(window_name, augmented_img.shape[1]//2, augmented_img.shape[0]//2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



