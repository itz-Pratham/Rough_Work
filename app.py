import streamlit as st
import cv2
import numpy as np
import face_recognition
from PIL import Image
import requests
from io import BytesIO
import datetime
import json

st.set_page_config(page_title="Interview Frame Validator", layout="centered")

# --- Helper Functions ---
def load_image_from_upload(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

def get_face_embeddings(image_np):
    rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    encodings = face_recognition.face_encodings(rgb)
    return encodings

def count_faces(image_np):
    rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    face_locations = face_recognition.face_locations(rgb)
    return len(face_locations)

def is_looking_at_screen(image_np):
    face_landmarks_list = face_recognition.face_landmarks(image_np)
    if not face_landmarks_list:
        return False
    for landmarks in face_landmarks_list:
        if "left_eye" in landmarks and "right_eye" in landmarks:
            return True
    return False

def frames_match(frame1_np, frame2_np):
    embeddings1 = get_face_embeddings(frame1_np)
    embeddings2 = get_face_embeddings(frame2_np)
    if not embeddings1 or not embeddings2:
        return False
    result = face_recognition.compare_faces([embeddings1[0]], embeddings2[0])
    return result[0]

def get_frame_input(label):
    st.subheader(f"📥 {label}")
    uploaded = st.file_uploader(f"Upload {label}", type=["jpg", "png", "jpeg"], key=f"upload_{label}")
    url = st.text_input(f"Or Enter URL for {label}", key=f"url_{label}")

    if uploaded:
        frame = load_image_from_upload(uploaded)
        st.image(frame, caption=f"{label} (Uploaded)", use_column_width=True)
        return frame
    elif url:
        try:
            frame = load_image_from_url(url)
            st.image(frame, caption=f"{label} (URL)", use_column_width=True)
            return frame
        except:
            st.error(f"❌ Failed to load {label} from URL.")
            st.stop()
    else:
        st.warning(f"⚠️ Please upload or enter URL for {label}.")
        st.stop()

# --- Main Logic ---
st.title("🎥 Interview Frame Validator")

frame1_np = get_frame_input("Frame 1")
frame2_np = get_frame_input("Frame 2")

# --- Processing ---
match = frames_match(frame1_np, frame2_np)
face_count_2 = count_faces(frame2_np)
looking_at_screen = is_looking_at_screen(frame2_np)

violations = 0
if not match:
    violations += 1
if face_count_2 != 1:
    violations += 1
if not looking_at_screen:
    violations += 1

result = {
    "timestamp": datetime.datetime.now().isoformat(),
    "matching_faces": int(match),
    "num_faces": face_count_2,
    "looking_at_screen": int(looking_at_screen),
    "violations_detected": int(violations > 0)
}

st.subheader("📊 Analysis Result")
st.json(result)
