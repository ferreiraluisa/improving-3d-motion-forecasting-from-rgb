import cv2
import os

img_folder = "vid"
imgs = sorted([f for f in os.listdir(img_folder) if f.endswith(".png")])

frame = cv2.imread(os.path.join(img_folder, imgs[0]))
h, w, _ = frame.shape

out = cv2.VideoWriter("output4.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h))

for img in imgs:
    out.write(cv2.imread(os.path.join(img_folder, img)))

out.release()