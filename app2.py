import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from deepface import DeepFace
import mediapipe as mp
import json

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Session state to store reference frame
if 'reference_frame' not in st.session_state:
    st.session_state.reference_frame = None

# Load image from file uploader
def load_image_from_upload(uploaded_file):
    image = Image.open(uploaded_file)
    return np.array(image)

# Load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(image)

# Count faces in image
def count_faces(image_np):
    results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return len(results.detections) if results.detections else 0

# Check if eyes are looking forward using landmarks
def is_looking_at_screen(image_np):
    results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        return 0
    landmarks = results.multi_face_landmarks[0].landmark
    left_eye_x = landmarks[33].x
    right_eye_x = landmarks[263].x
    mid = (left_eye_x + right_eye_x) / 2
    center_x = 0.5
    return 1 if abs(mid - center_x) < 0.1 else 0

# Match faces using DeepFace
def match_faces(reference_image, new_image):
    try:
        result = DeepFace.verify(reference_image, new_image, enforce_detection=False)
        return 1 if result['verified'] else 0
    except Exception as e:
        return 0

st.title("📷 Interview Frame Validator")

st.subheader("📥 Upload or Enter Image")
uploaded_image = st.file_uploader("Upload Frame Image", type=["jpg", "jpeg", "png"])
url = st.text_input("OR Enter Image URL")

if uploaded_image:
    current_frame = load_image_from_upload(uploaded_image)
    st.image(current_frame, caption="Uploaded Frame", use_column_width=True)
elif url:
    try:
        current_frame = load_image_from_url(url)
        st.image(current_frame, caption="Frame from URL", use_column_width=True)
    except:
        st.error("❌ Unable to fetch image from URL")
        st.stop()
else:
    st.info("Upload an image or enter a URL to proceed.")
    st.stop()

# If no reference frame is stored, use this one
if st.session_state.reference_frame is None:
    st.session_state.reference_frame = current_frame
    st.success("✅ First frame stored as reference. Upload next frame to compare.")
    st.stop()

# Evaluate new frame
num_faces = count_faces(current_frame)
matching_faces = match_faces(st.session_state.reference_frame, current_frame)
looking_at_screen = is_looking_at_screen(current_frame)
violations_detected = int(not (matching_faces and num_faces == 1 and looking_at_screen))

timestamp = str(st.session_state.get('timestamp', st.time()))

result = {
    "timestamp": timestamp,
    "matching_faces": matching_faces,
    "num_faces": num_faces,
    "looking_at_screen": looking_at_screen,
    "violations_detected": violations_detected
}

st.subheader("📊 Frame Analysis Result")
st.json(result)

# Option to reset reference frame
if st.button("🔄 Reset Reference Frame"):
    st.session_state.reference_frame = current_frame
    st.success("Reference frame updated.")
