# import cv2
import albumentations as A
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

def visiualize(image):
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()

def plot_examples(images):
    fig = plt.figure(figsize=(15,15))
    columns = 4
    rows = 5

    for i in range(1, len(images)):
        img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

# load image using cv2
# image = cv2.imread('dataset/playground_images/cat.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = Image.open('../dataset/playground_images/cat.jpg')

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5)
        ], p=1.0)
    ]
)

image_list = [image]
image = np.array(image)

for i in range(15):
    augmentations = transform(image=image)
    augmented_image = augmentations['image']
    image_list.append(augmented_image)

plot_examples(image_list)

if __name__ == "__main__":
    print("🇨🇦Albumentations Done")