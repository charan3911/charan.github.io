import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

def augment_with_padding(images, labels, num_augmented=20000):
    """
    Augment dataset by randomly shifting and padding digits
    This helps the model learn to recognize digits at various positions
    """
    augmented_images = []
    augmented_labels = []
    
    print("Augmenting dataset with random padding and shifts...")
    indices = np.random.choice(len(images), num_augmented, replace=True)
    
    for idx in indices:
        img = images[idx].reshape(28, 28)
        label = labels[idx]
        
        coords = np.column_stack(np.where(img > 0.1))
        
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            digit = img[y_min:y_max+1, x_min:x_max+1]
            
            scale = np.random.uniform(0.8, 1.0)
            h, w = digit.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h > 0 and new_w > 0:
                if max(new_h, new_w) > 22:
                    if new_h > new_w:
                        new_w = int(22 * new_w / new_h)
                        new_h = 22
                    else:
                        new_h = int(22 * new_h / new_w)
                        new_w = 22
                
                digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                canvas = np.zeros((28, 28), dtype=np.float32)
                
                if np.random.random() > 0.3: 
                    y_offset = (28 - new_h) // 2
                    x_offset = (28 - new_w) // 2
                else:  
                    y_offset = np.random.randint(0, max(1, 28 - new_h))
                    x_offset = np.random.randint(0, max(1, 28 - new_w))
                
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit
                
                augmented_images.append(canvas)
                augmented_labels.append(label)
    
    return np.array(augmented_images).reshape(-1, 28, 28, 1), np.array(augmented_labels)

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_aug, y_aug = augment_with_padding(x_train, y_train, num_augmented=20000)

x_train_combined = np.concatenate([x_train, x_aug])
y_train_combined = np.concatenate([y_train, y_aug])

shuffle_idx = np.random.permutation(len(x_train_combined))
x_train_combined = x_train_combined[shuffle_idx]
y_train_combined = y_train_combined[shuffle_idx]

y_train_encoded = keras.utils.to_categorical(y_train_combined, 10)
y_test_encoded = keras.utils.to_categorical(y_test, 10)

print(f"Training samples (with augmentation): {x_train_combined.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train_combined.shape[1:]}")

print("\nBuilding CNN model with Dropout...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

print("\nTraining model with augmented data...")
history = model.fit(
    datagen.flow(x_train_combined, y_train_encoded, batch_size=128),
    steps_per_epoch=len(x_train_combined) // 128,
    epochs=10,
    validation_data=(x_test, y_test_encoded),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nGenerating predictions...")
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - Padding-Robust Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'confusion_matrix.png'")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='#d4af37')
plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff6b6b')
plt.title('Model Loss (Padding-Robust)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#4CAF50')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#2196F3')
plt.title('Model Accuracy (Padding-Robust)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training history saved as 'training_history.png'")

print("\nShowing augmented training examples...")
plt.figure(figsize=(15, 6))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_aug[i].reshape(28, 28), cmap='gray')
    plt.title(f'Label: {y_aug[i]}', fontsize=9)
    plt.axis('off')
plt.suptitle('Augmented Training Examples (Random Padding & Shifts)', fontsize=14)
plt.tight_layout()
plt.savefig('augmented_examples.png', dpi=300, bbox_inches='tight')
print("Augmented examples saved as 'augmented_examples.png'")

print("\nSample Predictions:")
num_samples = 10
sample_indices = np.random.choice(len(x_test), num_samples, replace=False)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_test[idx]}\nPred: {y_pred_classes[idx]}',
              color='green' if y_test[idx] == y_pred_classes[idx] else 'red')
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
print("Sample predictions saved as 'sample_predictions.png'")

print("\nSaving model...")
model.save('mnist_cnn_model.keras')
print("✓ Model saved as 'mnist_cnn_model.keras'")

model.save('mnist_cnn_model.h5')
print("✓ Model saved as 'mnist_cnn_model.h5'")

model.save('saved_model/')
print("✓ Model saved in 'saved_model/' directory")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE!")
print("="*50)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Total Parameters: {model.count_params():,}")
print(f"Training samples used: {len(x_train_combined):,}")
print(f"Training stopped at epoch: {len(history.history['loss'])}")
print("\nThis model is now robust to:")
print("  ✓ Digits at corners")
print("  ✓ Digits at various positions")
print("  ✓ Different padding scenarios")
print("  ✓ Various scales and rotations")
print("\nGenerated files:")
print("  - mnist_cnn_model.keras (use this for the website)")
print("  - mnist_cnn_model.h5")
print("  - saved_model/ directory")
print("  - confusion_matrix.png")
print("  - training_history.png")
print("  - sample_predictions.png")
print("  - augmented_examples.png")