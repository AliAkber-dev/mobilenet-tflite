import tensorflow as tf
import cv2
import math
import numpy as np
def tf_cosine_similarity(embedding_1, embedding_2):
    cosine_similarity_loss = tf.keras.losses.CosineSimilarity(axis=1)
    cosine_similarity = -1 * cosine_similarity_loss(embedding_1, embedding_2).numpy()
    return round(cosine_similarity,3)

def calc_cosine_similarity(embedding_1, embedding_2):
    A = embedding_1[0]
    B = embedding_2[0]
    dot_product = 0 
    m_A = 0
    m_B = 0
    for idx, val in enumerate(A):
        dot_product += A[idx] * B[idx]
        m_A += A[idx] * A[idx]
        m_B += B[idx] * B[idx]
    m_A = math.sqrt(m_A)
    m_B = math.sqrt(m_B)
    similarity = dot_product/(m_A*m_B)
    return round(similarity,3)
    
def load_img(file_path=""):
    image_path = file_path  # Change this to the path of your image
    image = cv2.imread(image_path)

    # Preprocess the image (you may need to resize or normalize it depending on your model requirements)
    # For example, if your model expects images to be of a specific size:
    # resized_image = cv2.resize(image, (112, 112))  # Resize to 224x224 pixels
    normalized_image = image.astype(np.float32)
    normalized_image -= 127.5
    normalized_image /= 128.0

    resized_image = cv2.resize(normalized_image, (112, 112), interpolation=cv2.INTER_LINEAR)

    # Convert the image to a format suitable for TensorFlow
    image_for_tf = tf.convert_to_tensor(resized_image, dtype=tf.float32)
    image_for_tf = np.expand_dims(image_for_tf, axis=0)
    print(image_for_tf.shape)
    return image_for_tf

def get_embedding(model_tf, img_path=""):
    # Load the TFLite model and allocate tensors.
    model_tf.allocate_tensors()

    # Get input and output tensors.
    input_details = model_tf.get_input_details()
    output_details = model_tf.get_output_details()
    
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_pic = load_img(file_path=img_path)
    print("INPUT_SHAPE ALLOWED for model:", input_shape)
    input_data = input_pic
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    model_tf.set_tensor(input_details[0]['index'], input_data)

    model_tf.invoke()

    # # The function `get_tensor()` returns a copy of the tensor data.
    # # Use `tensor()` in order to get a pointer to the tensor.
    output_data = model_tf.get_tensor(output_details[0]['index'])
    return output_data
    # female_pic1 = load_img("../pics/female1-pic.png")
    # female_pic2 = load_img("../pics/female2-pic.png")


if __name__ == "__main__":
    interpreter = tf.lite.Interpreter(model_path="mobilenet/mobilefacenet.tflite")
    female1_pic1_path = "./pics/male1-pic1.png"
    embedding_1 = get_embedding(model_tf=interpreter, img_path=female1_pic1_path)
    female1_pic2_path = "./pics/male1-pic2.png" 
    embedding_2 = get_embedding(model_tf=interpreter, img_path=female1_pic2_path)
    cos_sim_tf = tf_cosine_similarity(embedding_1=embedding_1, embedding_2=embedding_2)
    print(cos_sim_tf)
    cos_sim_calc = calc_cosine_similarity(embedding_1=embedding_1, embedding_2=embedding_2)
    #This if else goes inside React Native code
    if cos_sim_calc > 0.6:
        print("Authentication Successful")
    else:
        print("Authentication Denied")