import cv2
import matplotlib.pyplot as plt


def debug_draw_pts(negative, pts, label=""):
    img = negative.get_debug_image()
    radius = 4
    color = (255, 0, 255)

    for p in pts:
        cv2.circle(img, p, radius, color, thickness=-1)

    if label != "":
        debug_draw_text(img, label)

    cv2.imshow("Debug", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Debug")


def debug_draw_text(image, text, text_orig=(100, 100)):
    color = (255, 0, 255)
    thickness = 1
    scale = 1
    cv2.putText(image, text, text_orig,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def debug_draw_rect(image, RectPts):
    color = (255, 0, 255)
    thickness = 1
    p1, p2 = RectPts
    cv2.rectangle(image, p1, p2, color, thickness)

    cv2.imshow("Debug", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Debug")


def get_user_defined_rect(image):
    global mouse_pos
    mouse_pos = list()
    cv2.imshow("Debug", image)
    cv2.setMouseCallback("Debug", on_mouse)

    while len(mouse_pos) < 2:
        cv2.waitKey(0)

    color = (255, 0, 255)
    thickness = 1
    p1 = mouse_pos.pop(0)
    p2 = mouse_pos.pop(0)
    cv2.rectangle(image, p1, p2, color, thickness)

    cv2.imshow("Debug", image)
    cv2.waitKey(0)
    cv2.destroyWindow("Debug")

    return p1, p2


def show_image_areas(areas, n):
    fig, axs = plt.subplots(1, n, sharey=True)
    i = 0
    for image_area in areas:
        axs[i].imshow(image_area)
        i += 1
    plt.show()


def on_mouse(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pos.append((x, y))
