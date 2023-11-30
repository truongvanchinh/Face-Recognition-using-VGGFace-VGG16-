import av
import cv2
import streamlit as st
import os
import mediapipe as mp
import numpy as np

from streamlit_webrtc import webrtc_streamer
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

@st.cache_resource
def load_facedetection():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    return face_detection

@st.cache_resource
def load_vggface_model():
    vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    for layer in vgg_model.layers:
        layer.trainable = False
    return vgg_model


def extract_feature_from_image(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Trích xuất đặc trưng từ ảnh
    feature = model.predict(x)
    
    # Reshape features từ (1, 512) về (512,)
    feature = np.squeeze(feature)
    return feature

# Hàm tính cosine similarity
def cosine_similarity(vector_1, vector_2):
    return np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))

# @st.cache_data
def load_known_embed_vectors(_model):
    dict = {}
    for name in os.listdir("Imgs"):
        dict[name] = []
        folder_path = os.path.join("Imgs", name)
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=(224, 224))    # Đọc ảnh và chỉnh về kích thước (224, 224)
            embedded_vector = extract_feature_from_image(img, _model)
            dict[name].append(embedded_vector)
    return dict

face_detection = load_facedetection()
vggface_model = load_vggface_model()
embedding_dict = load_known_embed_vectors(vggface_model)



##### GIAO DIỆN THÊM ẢNH #####
def add_face(face_detection):
    save_image = False  # Flag to take picture
    
    def callback(frame):
        nonlocal save_image
        nonlocal label
        nonlocal face_detection
        
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        results = face_detection.process(img)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Lấy ra hình chữ nhật của khuôn mặt
                face = img[y : y + height, x : x + width]
                cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Lưu ảnh khi nút "Take picture" được nhấn
        if save_image:
            if label:
                pass
            else:
                label = "Unknown"

            folder_path = os.path.join("Imgs", label)
            img_path = os.path.join(folder_path, "1.jpg")
            if os.path.isdir(folder_path):  # Nếu đã tồn tại folder ảnh của người đó
                # Lấy ra số tiếp theo để lưu vào
                next_number = len(os.listdir(folder_path)) + 1
                img_path = os.path.join(folder_path, f"{next_number}.jpg")
                cv2.imwrite(img_path, face)
                save_image = False
            else:
                os.makedirs(folder_path)
                cv2.imwrite(img_path, face)
                save_image = False
            
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")


    webrtc_streamer(key='camera',
                    video_frame_callback=callback,
                    sendback_audio=False)

    label = st.text_input("Enter your name: ", placeholder="Unknown")
    
    take_picture_button = st.button("Take picture")
    if take_picture_button:
        save_image = True
       



def label(embedded_vector, embedding_dict):
    threshold = 0.65
    max_cos_sim = -float('inf')
    for key in embedding_dict.keys():
        arr = []
        for vector in embedding_dict[key]:
            cos_sim = cosine_similarity(embedded_vector, vector)
            arr.append(cos_sim)
        avg = np.max(arr)
        if avg > max_cos_sim:
            label = key
            max_cos_sim = avg
            
    if max_cos_sim < threshold:
        label = "Unknown"
    return label
    


##### GIAO DIỆN PHÂN BIỆT KHUÔN MẶT #####
def face_recognition(face_detection, embedding_dict, model):
    def callback(frame):
        nonlocal face_detection
       
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        results = face_detection.process(img)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x = int(bbox.xmin * w)
                y =  int(bbox.ymin * h)
                
                width = int(bbox.width * w)
                height = int(bbox.height * h)

            
                
                # Lấy ra hình chữ nhật của khuôn mặt
                face = img[y : y + height, x : x + width]
                if (x>0 and y>0):
                    img = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    face_to_extract = cv2.resize(face, (224, 224))
                    embedding_vector = extract_feature_from_image(face_to_extract, model)
                    
                    if len(embedding_dict) == 0:
                        name = "Unknown"
                    else:
                        name = label(embedding_vector,  embedding_dict)
                    
                    img = cv2.putText(img=img,
                                    text=name,
                                    org=(x, y),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(255, 0, 0),
                                    thickness=2)
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    webrtc_streamer(key='camera',
                    video_frame_callback=callback,
                    sendback_audio=False)


########### INTERFACE ###########
# face_detection = load_facedetection()
# vggface_model = load_vggface_model()
# embedding_dict = load_known_embed_vectors(vggface_model)

genre = st.sidebar.radio(label='Mode', options=['Add face', 'Face recognition'])
if genre == "Add face":
    add_face(face_detection)
else:
    st.write("")
    face_recognition(face_detection, embedding_dict, vggface_model)