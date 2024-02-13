import tensorflow as tf

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(11,)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, X_train, y_train, epochs=20):
    history = model.fit(X_train, y_train, epochs=epochs)
    return history
