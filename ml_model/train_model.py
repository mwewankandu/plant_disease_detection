# ml_model/train_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class PlantDiseaseModel:
    def __init__(self, img_size=224, num_classes=3):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create a CNN model using MobileNetV2 (lightweight for mobile)"""
        # Using transfer learning with MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create new model on top
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        
        # Preprocessing (same as MobileNetV2)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Add new layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created successfully!")
        self.model.summary()
        return self.model
    
    def prepare_data(self, data_dir, batch_size=32):
        """Prepare data generators"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # Use 20% for validation
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Train generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Get class names
        self.class_names = list(train_generator.class_indices.keys())
        print(f"Classes detected: {self.class_names}")
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=10):
        """Train the model"""
        print("Starting training...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                save_best_only=True
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, validation_generator):
        """Evaluate model performance"""
        loss, accuracy = self.model.evaluate(validation_generator)
        print(f"Validation Accuracy: {accuracy*100:.2f}%")
        print(f"Validation Loss: {loss:.4f}")
        return accuracy, loss
    
    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100)
        plt.show()
    
    def save_model(self, path='plant_disease_model.h5'):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
        
        # Also save class names
        import pickle
        with open('class_names.pkl', 'wb') as f:
            pickle.dump(self.class_names, f)
    
    def convert_to_tflite(self):
        """Convert model to TensorFlow Lite for mobile"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("Model converted to TFLite format!")
        return tflite_model

def main():
    # Create model instance
    model = PlantDiseaseModel(img_size=224, num_classes=3)
    
    # Create model architecture
    model.create_model()
    
    # Prepare data (adjust path to your dataset)
    data_dir = 'dataset_mini'  # Change this to your dataset path
    
    # Create mini dataset if it doesn't exist
    if not os.path.exists(data_dir):
        print("Creating mini dataset structure...")
        os.makedirs(os.path.join(data_dir, 'train/Tomato_Early_Blight'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'train/Tomato_Healthy'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'train/Potato_Late_Blight'), exist_ok=True)
        print(f"Please add images to {data_dir}/train/ directories")
        return
    
    train_gen, val_gen = model.prepare_data(os.path.join(data_dir, 'train'))
    
    # Train model
    history = model.train(train_gen, val_gen, epochs=15)
    
    # Evaluate
    accuracy, loss = model.evaluate(val_gen)
    
    # Plot training history
    model.plot_history()
    
    # Save model
    model.save_model()
    
    # Convert to TFLite for mobile
    model.convert_to_tflite()
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE!")
    print(f"Final Validation Accuracy: {accuracy*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()