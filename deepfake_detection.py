import os
import numpy as np
import cv2
import tensorflow as tf
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

    # Model architecture
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
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

def train_model(model, train_images, train_labels, val_images, val_labels, epochs=15):
    """Enhanced training with callbacks and monitoring"""
    # Calculate precise class weights
    if len(np.unique(train_labels)) > 1:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weights}")
    else:
        class_weights = None

    # Enhanced data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        train_datagen.flow(train_images, train_labels, batch_size=32),
        validation_data=val_datagen.flow(val_images, val_labels),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    return history

def evaluate_model(model, test_images, test_labels):
    """Comprehensive model evaluation"""
    print("\nEvaluating model...")
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(
        test_images, test_labels, verbose=0
    )
    
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Detailed classification report
    predictions = model.predict(test_images)
    predicted_classes = (predictions > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_classes, target_names=['Real', 'Fake']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, predicted_classes))
    
    return test_acc

if __name__ == "__main__":
    dataset_path = 'D:/Notes/Degree/Projects/DEEP FAKE/Data'

    # Load Data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_dataset(dataset_path)

    # Create Model
    model = create_model()
    model.summary()

    # Train Model
    print("\nStarting training...")
    train_model(model, train_images, train_labels, val_images, val_labels, epochs=15)

    # Evaluate Model
    test_accuracy = evaluate_model(model, test_images, test_labels)

    # Save Model
    if test_accuracy > 0.75:  # Only save if accuracy is reasonable
        model.save('deepfake_detection_model_enhanced.h5')
        print("Model saved successfully.")
    else:
        print("Model accuracy too low, not saving.")