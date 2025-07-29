def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train a machine learning model.

    Parameters:
    model: The machine learning model to be trained.
    X_train: Training features.
    y_train: Training labels.
    X_val: Validation features.
    y_val: Validation labels.
    epochs: Number of epochs for training.
    batch_size: Size of each batch during training.

    Returns:
    history: Training history containing loss and accuracy metrics.
    """
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=epochs, 
                        batch_size=batch_size)
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.

    Parameters:
    model: The trained machine learning model.
    X_test: Test features.
    y_test: Test labels.

    Returns:
    metrics: A dictionary containing evaluation metrics.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics


def save_model(model, filepath):
    """
    Save the trained model to a file.

    Parameters:
    model: The trained machine learning model.
    filepath: The path where the model will be saved.
    """
    model.save(filepath)