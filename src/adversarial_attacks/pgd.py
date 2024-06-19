import tensorflow as tf

def pgd_attack(model, images, labels, epsilon, alpha, num_iterations):
    # Ensure images and labels are tensors
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    
    perturbed_images = images
    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(perturbed_images)
            prediction = model(perturbed_images)
            loss = tf.keras.losses.categorical_crossentropy(labels, prediction)
        
        gradient = tape.gradient(loss, perturbed_images)
        perturbation = alpha * tf.sign(gradient)
        perturbed_images = perturbed_images + perturbation
        perturbation = tf.clip_by_value(perturbed_images - images, -epsilon, epsilon)
        perturbed_images = tf.clip_by_value(images + perturbation, 0, 1)
    return perturbed_images

# import tensorflow as tf

# def pgd_attack(model, images, labels, epsilon, alpha, num_iterations):
#     # Ensure images and labels are tensors
#     images = tf.convert_to_tensor(images)
#     labels = tf.convert_to_tensor(labels)
    
#     perturbed_images = images
#     for _ in range(num_iterations):
#         with tf.GradientTape() as tape:
#             tape.watch(perturbed_images)
#             prediction = model(perturbed_images)
#             loss = tf.keras.losses.categorical_crossentropy(labels, prediction)
        
#         gradient = tape.gradient(loss, perturbed_images)
#         perturbation = alpha * tf.sign(gradient)
#         perturbed_images = perturbed_images + perturbation
#         perturbation = tf.clip_by_value(perturbed_images - images, -epsilon, epsilon)
#         perturbed_images = tf.clip_by_value(images + perturbation, 0, 1)
#     return perturbed_images


# # import tensorflow as tf

# # def pgd_attack(model, image, label, epsilon, alpha, num_iterations):
# #     perturbed_image = image
# #     for _ in range(num_iterations):
# #         with tf.GradientTape() as tape:
# #             tape.watch(perturbed_image)
# #             prediction = model(perturbed_image)
# #             loss = tf.keras.losses.categorical_crossentropy(label, prediction)
        
# #         gradient = tape.gradient(loss, perturbed_image)
# #         perturbation = alpha * tf.sign(gradient)
# #         perturbed_image = perturbed_image + perturbation
# #         perturbation = tf.clip_by_value(perturbed_image - image, -epsilon, epsilon)
# #         perturbed_image = tf.clip_by_value(image + perturbation, 0, 1)
# #     return perturbed_image

