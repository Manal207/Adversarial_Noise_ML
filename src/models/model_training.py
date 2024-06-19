import tensorflow as tf

def train_model(model, train_images, train_labels, validation_data, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    return history
