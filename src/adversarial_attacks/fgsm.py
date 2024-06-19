import tensorflow as tf

def fgsm_attack(model, images, labels, epsilon):
    # Ensure images and labels are tensors
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    
    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)
        loss = tf.keras.losses.categorical_crossentropy(labels, prediction)
    gradient = tape.gradient(loss, images)
    perturbation = epsilon * tf.sign(gradient)
    adversarial_images = images + perturbation
    adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
    return adversarial_images

# import tensorflow as tf

# def fgsm_attack(model, image, label, epsilon):
#     with tf.GradientTape() as tape:
#         tape.watch(image)
#         prediction = model(image)
#         loss = tf.keras.losses.categorical_crossentropy(label, prediction)
#     gradient = tape.gradient(loss, image)
#     perturbation = epsilon * tf.sign(gradient)
#     adversarial_image = image + perturbation
#     adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
#     return adversarial_image


