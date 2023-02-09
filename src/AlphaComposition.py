import cv2
import numpy as np

def alpha_compose(img, num_squares, alpha):
    height, width, channels = img.shape
    img_copy = img.copy()
    for i in range(num_squares):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        size = np.random.randint(10, min(width//3, height//3))
        x2 = x1 + size
        y2 = y1 + size
        if x2 > width:
            x2 = width
            x1 = x2 - size
        if y2 > height:
            y2 = height
            y1 = y2 - size
        square = np.ones((size, size, channels), dtype=np.float32) * 0
        square = cv2.resize(square, (y2-y1, x2-x1), interpolation = cv2.INTER_NEAREST)
        img_copy[y1:y2, x1:x2] = square * alpha + img_copy[y1:y2, x1:x2] * (1 - alpha)
    return img_copy


if __name__ == '__main__':
    image_path = "../MyImage.jpg"
    img = cv2.imread(image_path)
    num_squares = 5
    alpha = 0.6
    augmented_img = alpha_compose(img, num_squares, alpha)

    both_images = np.hstack((img, augmented_img))
    cv2.namedWindow("Both Images", cv2.WINDOW_NORMAL)
    cv2.imshow("Both Images", both_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





