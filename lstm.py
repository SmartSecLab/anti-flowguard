import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


def plot_metrics(history, cm):
    """ Plot the metrics of the model """
    # Plotting all in one figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # Plot training & validation accuracy values
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()

    # Plot training & validation loss values
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                cbar=False, ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xlabel("Predicted Label")
    axes[1, 0].set_ylabel("True Label")

    # Plot precision
    axes[1, 1].plot(history.history['Precision'], label='Val Precision')
    axes[1, 1].set_title('Validation Precision')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].legend()

    # Plot recall
    axes[2, 0].plot(history.history['Recall'], label='Val Recall')
    axes[2, 0].set_title('Validation Recall')
    axes[2, 0].set_xlabel('Epochs')
    axes[2, 0].set_ylabel('Recall')
    axes[2, 0].legend()

    # Plot F1-score
    axes[2, 1].plot(history.history['f1_score'], label='Val F1-Score')
    axes[2, 1].set_title('Validation F1-Score')
    axes[2, 1].set_xlabel('Epochs')
    axes[2, 1].set_ylabel('F1-Score')
    axes[2, 1].legend()

    plt.tight_layout()
    plt.show()
    return fig


def train_LSTM(X, y):
    """ Train an LSTM model on the given data and return the trained model, history and figure of metrics """

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Convert DataFrames to NumPy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # Reshaping features for LSTM input
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Building the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, X_train.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
                  'accuracy', 'Precision', 'Recall', 'f1_score'])

    # Training the model
    history = model.fit(X_train, y_train, epochs=10,
                        batch_size=32, validation_data=(X_test, y_test))

    # Evaluating the model
    loss, accuracy, precision, recall, f1_score = model.evaluate(
        X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print(f"Test F1_score: {f1_score}")

    # Predicting on the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Print confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return model, history, cm


def apply_lstm(X, y):
    """ Apply LSTM model on the given data and return the trained model, history and figure of metrics """
    # Train the LSTM model
    model, history, cm = train_LSTM(X, y)

    # Plotting the metrics
    fig = plot_metrics(history, cm)

    # save the plt figure
    fig.savefig('figure/lstm_model_metrics.png')

    # Save the model
    model.save('models/lstm_model.h5')

    # Save the history to a text file
    with open('models/LSTM-history.txt', 'w') as f:
        f.write(str(history.history))


# Apply the LSTM model
# apply_LSTM(X, y)
