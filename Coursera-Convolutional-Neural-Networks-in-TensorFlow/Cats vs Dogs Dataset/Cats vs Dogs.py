import os
from pickletools import optimize
import random
import shutil
import NerualNetwork.helloWorld as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt


def main():
    source_dir = os.path.join(os.path.dirname(__file__), "PetImages") 
    source_cats_dir = os.path.join(source_dir, "Cats")
    source_dogs_dir = os.path.join(source_dir, "Dogs")
    tat_dir = os.path.join(os.path.dirname(__file__), "TrainAndTest") 
    if os.path.exists(tat_dir):
        shutil.rmtree(tat_dir)
    os.makedirs(tat_dir)
    train_dir = os.path.join(tat_dir, "Train")
    os.makedirs(train_dir)
    test_dir = os.path.join(tat_dir, "Test")
    os.makedirs(test_dir)
    train_cats_dir = os.path.join(train_dir, "Cats")
    train_dogs_dir = os.path.join(train_dir, "Dogs")
    os.makedirs(train_cats_dir)
    os.makedirs(train_dogs_dir)
    test_cats_dir = os.path.join(test_dir, "Cats")
    test_dogs_dir = os.path.join(test_dir, "Dogs")
    os.makedirs(test_cats_dir)
    os.makedirs(test_dogs_dir)

    def arrange(test_rate: float, SOURCE_PATH, TEST_PATH, TRAIN_PATH):
        files = os.listdir(SOURCE_PATH)
        files = [file for file in files if os.path.getsize(os.path.join(SOURCE_PATH, file)) != 0]
        files = random.sample(files, len(files))
        test_size = test_rate * len(files)
        for i in range(len(files)):
            shutil.copyfile(os.path.join(SOURCE_PATH, files[i]), os.path.join((TEST_PATH if i < test_size else TRAIN_PATH), files[i]))

    arrange(0.1, source_cats_dir, test_cats_dir, train_cats_dir)
    arrange(0.1, source_dogs_dir, test_dogs_dir, train_dogs_dir)

    train_generate = ImageDataGenerator(1./255.).flow_from_directory(
        directory=train_dir,
        batch_size=40,
        class_mode="binary",
        target_size=(150, 150)
    )

    test_generate = ImageDataGenerator(1./255.).flow_from_directory(
        directory=test_dir,
        batch_size=40,
        class_mode="binary",
        target_size=(150, 150)
    )

    model = tf.keras.models.Sequential([
        Conv2D(64, (3, 3), input_shape=(150, 150, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        losses = tf.keras.mea,
        metrics=(["accuracy"])
    )

    history = model.fit(
        train_generate,
        epochs=15,
        verbose=1,
        validation_data=test_generate
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.show()
    print("")

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.show()

if __name__ == "__main__":
    main()
