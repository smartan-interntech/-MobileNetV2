import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# === Define your class names ===
class_names = ['correct', 'incorrect']  # Update as per your actual labels

# === Load model ===
model = load_model("mobilenet_posture_classifier.h5")

# === Load test data ===
x_test = np.load("x_test.npy")  # shape: (N, 224, 224, 3)
y_test = np.load("y_test.npy")  # shape: (N,) or (N, num_classes)

# === Handle y_test format ===
if len(y_test.shape) == 1:
    y_true = y_test  # already integer encoded
else:
    y_true = np.argmax(y_test, axis=1)  # one-hot encoded

# === Predict ===
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# === Classification Report ===
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
