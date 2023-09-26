import matplotlib.pyplot as plt
import cv2


def visualise(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def plot_examples(images, bboxes=None):
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 5

    for i in range(1, len(images)):
        if bboxes is not None:
            img = visualise_bbox(images[i-1], bboxes[i-1], "sample")
        else:
            img = images[i-1]  # get current image in images list

        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def visualise_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=4):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    img = cv2.putText(img, class_name, (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return img
