import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):
    print("File path:", file_path)

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

import os

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the class dictionary
    class_dict_path = os.path.join(script_dir, 'artifacts', 'class_dictionary.json')

    with open(class_dict_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        model_path = os.path.join(script_dir, 'artifacts', 'saved_model.pkl')
        with open(model_path, 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")



def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_cascade_path = os.path.join(script_dir, 'opencv', 'haarcascades', 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(script_dir, 'opencv', 'haarcascades', 'haarcascade_eye.xml')

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)


    if image_path:
        img = cv2.imread(image_path)
        print("Image read successfully")
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
        print("Image from base64 successfully")
    if img is None:
        print("Image is None")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_virat():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    b64_file_path = os.path.join(script_dir, "b64.txt")

    with open(b64_file_path) as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
    # Add this code to check if the image file can be opened with OpenCV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_test_path = os.path.join(script_dir, "test_images", "federer1.jpg")
    img_test = cv2.imread(img_test_path)

    if img_test is not None:
        print("Test image read successfully.")
    else:
        print("Test image cannot be read.")

    

    # print(classify_image(get_b64_test_image_for_virat(), None))

    print(classify_image(None, img_test_path))
    # print(classify_image(None, "./test_images/federer2.jpg"))
    # print(classify_image(None, "./test_images/virat1.jpg"))
    # print(classify_image(None, "./test_images/virat2.jpg"))
    # print(classify_image(None, "./test_images/virat3.jpg")) # Inconsistent result could be due to https://github.com/scikit-learn/scikit-learn/issues/13211
    # print(classify_image(None, "./test_images/serena1.jpg"))
    # print(classify_image(None, "./test_images/serena2.jpg"))
    # print(classify_image(None, "./test_images/sharapova1.jpg"))
