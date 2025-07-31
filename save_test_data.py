# save_test_data.py
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

test_dir = "dataset/"  # adjust if needed
labels_map = {"correct": 0, "incorrect": 1}  # adjust based on your class names
x_data = []
y_data = []

for label in os.listdir(test_dir):
    label_path = os.path.join(test_dir, label)
    if not os.path.isdir(label_path): continue

    for img_file in tqdm(os.listdir(label_path)):
        img_path = os.path.join(label_path, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        x_data.append(img_array)
        y_data.append(labels_map[label])

x_data = np.array(x_data)
y_data = np.array(y_data)

np.save("x_test.npy", x_data)
np.save("y_test.npy", y_data)

print("✅ Saved x_test.npy and y_test.npy")
