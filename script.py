import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense, ReLU, Dropout, Softmax, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.regularizers import l2  # Import the L2 regularizer
from tensorflow.keras.optimizers import Adam

def load_data():
    X = np.load('X_cifar_grayscale.npy')
    Y = np.load('Y_cifar.npy')
    return X, Y

def preprocess_data(X):
    nb_features = 32 * 32
    X_flat = np.reshape(X, (X.shape[0], nb_features))
    return X_flat

def apply_pca(X_flat, n_components_list):
    X_pca_list = []
    for n_components in n_components_list:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_flat)
        X_pca_list.append(X_pca)
        visualize_reconstruction(pca, X_flat, X_pca, n_components)
    return X_pca_list

def visualize_reconstruction(pca, X_flat, X_pca, n_components):
    X_reconstructed = pca.inverse_transform(X_pca)
    plt.figure()
    plt.imshow(np.reshape(X_reconstructed[122], (32, 32)), cmap='gray')
    plt.title(f'Reconstructed with {n_components} components')
    plt.show()

def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test

def build_dense_model(input_shape, num_classes):
    
    inputs = Input(shape=input_shape)
    
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.002))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.002))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def build_cnn_model(input_shape, num_classes):
    
    input_layer = Input(shape=input_shape)
    x = Conv2D(4, (3, 3), padding='same')(input_layer)
    x = ReLU()(x)
    
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = ReLU()(x)
    
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = ReLU()(x)
    
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    
    x = Dense(64)(x)
    x = ReLU()(x)
    
    regularization_factor = 0.02
    output_layer = Dense(num_classes, activation='softmax', kernel_regularizer=l2(regularization_factor))(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, epochs=20, batch_size=100):
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')
    plot_training_history(history)

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# Main Execution
X, Y = load_data()
X_flat = preprocess_data(X)
n_components_list = [625, 400, 225, 100, 36]
X_pca_list = apply_pca(X_flat, n_components_list)
X_train, X_test, Y_train, Y_test = split_data(X_pca_list[1], Y)
X_train_cnn, X_test_cnn, Y_train_cnn, Y_test_cnn = train_test_split(X, Y, test_size=0.2)

# Train Dense Model
dense_model = build_dense_model(X_pca_list[1].shape[1], num_classes=3)
train_and_evaluate(dense_model, X_train, Y_train, X_test, Y_test)


# Train CNN Model
cnn_model = build_cnn_model((32, 32, 1), num_classes=3)
train_and_evaluate(cnn_model, X_train_cnn.reshape(-1, 32, 32, 1), Y_train_cnn, X_test_cnn.reshape(-1, 32, 32, 1), Y_test_cnn, epochs=8, batch_size=100)
