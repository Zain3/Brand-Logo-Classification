import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')

import tensorflow as tf
from tensorflow import keras

tf.keras.utils.set_random_seed(
    seed=0
)

def train_function(raw_data, raw_labels):

    from sklearn.model_selection import train_test_split

    # Training and Test sets
    X_training, X_test, t_training, t_test = train_test_split(X_train_full, 
                                                    t_train_full, 
                                                    shuffle=True,
                                                    stratify=t_train_full,
                                                    test_size=0.15)
    # Train and validation sets
    X_train, X_val, t_train, t_val = train_test_split(X_training, 
                                                    t_training, 
                                                    shuffle=True,
                                                    stratify=t_training,
                                                    test_size=0.2)

    X_training.shape, t_training.shape, X_train.shape, t_train.shape, X_val.shape, t_val.shape


    X_training_reshaped = X_training.reshape(-1, 300, 300, 3)
    X_test_reshaped = X_test.reshape(-1, 300, 300, 3)

    # Reshape the input data to match the model's expected input shape
    X_train_reshaped = X_train.reshape(-1, 300, 300, 3)
    X_val_reshaped = X_val.reshape(-1, 300, 300, 3)


    data_augmentation = tf.keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomBrightness(0.3),
    keras.layers.RandomContrast(0.4),
    #keras.layers.RandomCrop(height=0.5,width=0.5,seed=0),
    keras.layers.RandomZoom(height_factor=0.5,width_factor=0.5,seed=0),
    #keras.layers.RandomWidth(),
    ])


    X_train_sample = X_train_reshaped
    t_train_sample = t_train

    t_train_append = np.append(t_train_sample,t_train_sample)
    t_train_append = np.append(t_train_append,t_train_sample)
    t_train_append = np.append(t_train_append,t_train_sample)
    t_train_augmented = np.append(t_train_append,t_train_sample)


    def return_augmented_dataset(dataset):
        augmented_dataset =data_augmentation(dataset)
        augmented_dataset_numpy = augmented_dataset.numpy()
        return augmented_dataset_numpy


    augmented_dataset1 = return_augmented_dataset(X_train_sample)
    augmented_dataset2 = return_augmented_dataset(X_train_sample)
    augmented_dataset3 = return_augmented_dataset(X_train_sample)
    augmented_dataset4 = return_augmented_dataset(X_train_sample)

    augmented_dataset = np.append(X_train_reshaped,augmented_dataset1, axis=0)
    augmented_dataset = np.append(augmented_dataset,augmented_dataset2, axis=0)
    augmented_dataset = np.append(augmented_dataset,augmented_dataset3, axis=0)
    X_train_augmented = np.append(augmented_dataset,augmented_dataset4, axis=0)


    base_model = keras.applications.resnet50.ResNet50(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False)  # Do not include the ImageNet classifier at the top.


    # Freeze base model
    base_model.trainable = False


    model_seq = tf.keras.Sequential()
    model_seq.add(keras.layers.Dropout(0.25))
    model_seq.add(base_model)
    model_seq.add(keras.layers.GlobalAveragePooling2D())
    model_seq.add(keras.layers.Dropout(0.50))
    model_seq.add(keras.layers.Dense(512, activation='relu'))
    model_seq.add(keras.layers.Dropout(0.50))
    model_seq.add(keras.layers.Dense(256, activation='relu'))
    model_seq.add(keras.layers.Dropout(0.50))
    model_seq.add(keras.layers.Dense(128, activation='relu'))


    IMG_SIZE = 150
    # .Input() instantiates a Keras tensor
    inputs = keras.Input(shape=(300, 300, 3))
    # Input layer
    inputs_resized = tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE)(inputs)
    # resizing input to match pretrained model
    x = model_seq(inputs_resized, training=False)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.


    # Flattening
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x_flatten = keras.layers.Flatten()(x)
    x_flatten.shape


    outputs = keras.layers.Dense(10, activation='softmax')(x_flatten)
    model = keras.Model(inputs, outputs)


    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                optimizer=keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                metrics=['accuracy'])

    model.fit(X_train_augmented,t_train_augmented, epochs=15, batch_size=32,
            validation_data=(X_val_reshaped, t_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)])


    model.save('final_model.h5')

#---------------------

X_train_full = np.load('data_train.npy').T
t_train_full = np.load('labels_train_corrected.npy')

train_function(X_train_full, t_train_full)