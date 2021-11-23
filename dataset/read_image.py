import cv2
def read_image(file_name):
    img=cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    img=img/256
    return img