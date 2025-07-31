# 📌 Posture Analysis using MobileNetV2 — Front Raises (Model 2)

## 🧠 Objective

To develop a **lightweight binary classifier** using **MobileNetV2** that can classify individual video frames of a gym exercise (*Front Raises*) as either **correct** or **incorrect** posture.

---

## 📁 Dataset Overview

* **Exercise Focus:** Front Raises

* **Total Frames Used:**

  * ✅ Correct: **617 frames**
  * ❌ Incorrect: **603 frames**

* **Video Source:**

  * Frames were extracted from labeled videos.
  * Videos split approximately 50–50 across correct and incorrect classes to ensure balance.

* **Input Resolution:** `224x224`

* **Batch Size:** `32`

* **Epochs Trained:** `50`

---

## 🔄 Data Preprocessing & Augmentation

### ✅ **Training Set Augmentation:**

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode='nearest'
)
```

* Used to simulate real-world variation like camera shake, lighting, and orientation.

### 📉 **Validation Set:**

```python
val_datagen = ImageDataGenerator(rescale=1./255)
```

* **No augmentation** applied. Purely used for unbiased model validation.

---

## 🏗️ Model Architecture

* **Base Model:** Pretrained `MobileNetV2` (`include_top=False`)
* **Layers:**

  * `GlobalAveragePooling2D`
  * `Dense(128, activation='relu')`
  * `Dropout(0.5)`
  * `Dense(1, activation='sigmoid')`  ← Binary classification
* **Loss Function:** `binary_crossentropy`
* **Optimizer:** `Adam`
* **Metrics:** `accuracy`

All base layers were **frozen** during training to leverage pre-trained weights.

---

## 📊 Results & Observations

* ✅ **Training Accuracy:** Good convergence with augmentation.
* 🤏 **Improved performance** over the Siamese model.
* 👤 **Performs well on known individuals (trainer data)**.
* ❌ **Fails to generalize** to unseen individuals:

  * Misclassifies frames
  * Predicts **“correct”** even for clearly incorrect posture
  * Outputs lack **confidence variation**

---

## 🚨 Key Issue Identified

> **On testing with new unseen videos, the model consistently predicts "correct" for all frames, even when the posture is incorrect.**

### Possible Reasons:

* Lack of **diverse subjects** in training data
* Model has possibly **overfit** on trainer's body posture or background
* No domain-specific postural constraints are enforced
* Data labeling might not reflect nuanced biomechanical errors

---



## 📎 Folder Structure (Sample)

```
bicepcurls/
├── Bicep_curl_log_20250729_103303.txt
├── Bicep_curl_log_20250729_165008.txt
├── bicep_curls_byuploadingfiles.py
├── bicep_curls_rtsp.py
├── output_bicep_curls1.mp4
├── output_bicep_curls2.mp4
├── yolov8n-pose.pt

frontraises/
├── dataset_split/
│   ├── train/
│   │   ├── correct/
│   │   └── incorrect/
│   └── val/
│       ├── correct/
│       └── incorrect/
├── augment.py
├── frames_convert.py
├── Front_raises_log_20250729_122738.txt
├── Front_raises_log_20250729_122829.txt
├── Front_raises_log_20250729_122917.txt
├── Front_raises_log_20250729_141132.txt
├── Front_raises_log_20250729_165449.txt
├── Frontraise_model.h5
├── Frontraise_prediction_log.txt
├── frontraises.py
├── mobilenet_posture_classifier.h5
├── output_Front_raises.mp4
├── output_frontraise.mp4
├── output_predicted_video.mp4
├── save_test_data.py
├── split.py
├── test.py
├── train.py
├── visualize.py
├── x_test.npy
├── y_test.npy
├── yolov8n-pose.pt

venv/

```

---

