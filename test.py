import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Enable GPU (CUDA) if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU detected: {physical_devices}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found. Running on CPU.")

def preprocess_image(image_path):
    """Enhanced image preprocessing"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MobileNetV2 expects RGB
    img = img / 255.0
    return img

def load_dataset(base_dir):
    """Improved dataset loading with error handling"""
    def load_images_from_folder(folder):
        images, labels = [], []
        for label, category in enumerate(['real', 'fake']):
            category_dir = os.path.join(folder, category)
            if not os.path.exists(category_dir):
                continue
                
            for file in os.listdir(category_dir):
                image_path = os.path.join(category_dir, file)
                img = preprocess_image(image_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
        return np.array(images), np.array(labels)

    print("Loading datasets...")
    train_images, train_labels = load_images_from_folder(os.path.join(base_dir, 'train'))
    val_images, val_labels = load_images_from_folder(os.path.join(base_dir, 'validation'))
    test_images, test_labels = load_images_from_folder(os.path.join(base_dir, 'test'))
    
    print(f"Dataset sizes - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    print(f"Class distribution - Train: {np.bincount(train_labels)}, Val: {np.bincount(val_labels)}, Test: {np.bincount(test_labels)}")
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def create_model(input_shape=(224, 224, 3)):
    """Enhanced model architecture"""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base layers

    model = models.Sequential([
        base_model,
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    return model

def plot_training_history(history):
    """Visualizing training progress"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

def visualize_confusion_matrix(y_true, y_pred):
    """Display confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def show_sample_predictions(model, test_images, test_labels):
    """Display sample test images with predictions"""
    predictions = (model.predict(test_images) > 0.5).astype(int)
    plt.figure(figsize=(10, 10))
    
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_images[i])
        plt.title(f"Actual: {'Real' if test_labels[i] == 0 else 'Fake'}\nPredicted: {'Real' if predictions[i] == 0 else 'Fake'}")
        plt.axis('off')
    
    plt.show()

def train_model(model, train_images, train_labels, val_images, val_labels, epochs=15):
    """Enhanced training with callbacks and monitoring"""
    if len(np.unique(train_labels)) > 1:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights)}
    else:
        class_weights = None
    
    train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                       shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    val_datagen = ImageDataGenerator()

    history = model.fit(
        train_datagen.flow(train_images, train_labels, batch_size=32),
        validation_data=val_datagen.flow(val_images, val_labels),
        epochs=epochs,
        class_weight=class_weights
    )
    plot_training_history(history)
    return history

def evaluate_model(model, test_images, test_labels):
    """Comprehensive model evaluation"""
    predictions = (model.predict(test_images) > 0.5).astype(int)
    visualize_confusion_matrix(test_labels, predictions)
    show_sample_predictions(model, test_images, test_labels)

if __name__ == "__main__":
    dataset_path = 'D:/Notes/Degree/Projects/DEEP FAKE/Data'
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_dataset(dataset_path)
    model = create_model()
    model.summary()
    train_model(model, train_images, train_labels, val_images, val_labels, epochs=15)
    evaluate_model(model, test_images, test_labels)

    # Save the trained model
    model.save('test_detection_model.h5')
    print("Model saved successfully.")