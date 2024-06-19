# import tensorflow as tf
# from cleverhans.tf2.attacks import jsma 

# def jsma_attack(model, image, label, theta=1.0, gamma=0.1):
#     adversarial_image = jsma(model, image, label, theta=theta, gamma=gamma)
#     adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
#     return adversarial_image

