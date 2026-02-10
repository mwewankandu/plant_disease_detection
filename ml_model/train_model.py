import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)


class PlantDiseaseModel:
    def __init__(self, img_size=224, num_classes=3):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = None

    # --------------------------------------------------
    # MODEL CREATION
    # --------------------------------------------------
    def create_model(self):
        """Create MobileNetV2-based classifier"""

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights="imagenet"
        )

        # Freeze base model initially
        base_model.trainable = False

        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))

        # IMPORTANT: same preprocessing everywhere ([-1, 1])
        x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)

        x = base_model(x, training=False)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("Model created successfully")
        self.model.summary()
        return self.model

    # --------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------
    def prepare_data(self, data_dir, batch_size=32):
        """Prepare generators with strong augmentation"""

        preprocess_fn = lambda x: (x / 127.5) - 1.0

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            validation_split=0.2
        )

        train_gen = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True
        )

        val_gen = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False
        )

        self.class_names = list(train_gen.class_indices.keys())
        print("Classes:", self.class_names)

        # Compute class weights (handles imbalance)
        labels = train_gen.classes
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )
        class_weights = dict(enumerate(class_weights))

        return train_gen, val_gen, class_weights

    # --------------------------------------------------
    # TRAINING PHASE 1 (FEATURE EXTRACTION)
    # --------------------------------------------------
    def train(self, train_gen, val_gen, class_weights, epochs=20):
        print("Phase 1: Training classifier head")

        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3),
            keras.callbacks.ModelCheckpoint(
                "best_model.keras",
                save_best_only=True
            )
        ]

        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    # --------------------------------------------------
    # TRAINING PHASE 2 (FINE-TUNING)
    # --------------------------------------------------
    def fine_tune(self, train_gen, val_gen, class_weights, fine_tune_at=-30, epochs=10):
        print("Phase 2: Fine-tuning MobileNetV2")

        base_model = self.model.layers[2]
        base_model.trainable = True

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history_fine = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            verbose=1
        )

        return history_fine

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------
    def evaluate(self, val_gen):
        loss, acc = self.model.evaluate(val_gen)
        print(f"Validation Accuracy: {acc * 100:.2f}%")
        return loss, acc

    def confusion_matrix_report(self, val_gen):
        preds = self.model.predict(val_gen)
        y_pred = np.argmax(preds, axis=1)
        y_true = val_gen.classes

        print("Classification Report")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    # --------------------------------------------------
    # SAVE + TFLITE
    # --------------------------------------------------
    def save_model(self):
        self.model.save("plant_disease_model.keras")
        with open("class_names.pkl", "wb") as f:
            pickle.dump(self.class_names, f)
        print("Model saved")

    def convert_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open("model.tflite", "wb") as f:
            f.write(tflite_model)

        print("TFLite model saved")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    data_dir = "dataset_mini/train"

    model = PlantDiseaseModel(img_size=224, num_classes=3)
    model.create_model()

    train_gen, val_gen, class_weights = model.prepare_data(data_dir)

    model.train(train_gen, val_gen, class_weights, epochs=20)
    model.fine_tune(train_gen, val_gen, class_weights, epochs=10)

    model.evaluate(val_gen)
    model.confusion_matrix_report(val_gen)

    model.save_model()
    model.convert_to_tflite()

    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
