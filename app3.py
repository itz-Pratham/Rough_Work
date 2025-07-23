# app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from deepface import DeepFace
import mediapipe as mp
from datetime import datetime
from typing import Optional, Dict, Tuple, List
import math
import warnings
warnings.filterwarnings("ignore")

# ----------------- CONSTANTS -----------------
DETECTOR_BACKEND = "retinaface"   # For detection/alignment
EMBED_MODEL      = "ArcFace"      # For embeddings
COSINE_THRESH    = 0.38           # Slightly lenient so we don't miss same person
GAZE_TOL         = 0.12           # Iris offset tolerance
YAW_MAX_DEG      = 15.0           # Head-pose fallback thresholds
PITCH_MAX_DEG    = 20.0
# ---------------------------------------------

st.set_page_config(page_title="Interview Frame Validator", layout="wide")

# ----------------- CACHED RESOURCES -----------------
@st.cache_resource
def load_mediapipe():
    mp_fd = mp.solutions.face_detection
    mp_fm = mp.solutions.face_mesh
    fd = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    fm = mp_fm.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    return fd, fm

@st.cache_resource
def load_arcface_model():
    return DeepFace.build_model(EMBED_MODEL)

face_detector_mp, face_mesh = load_mediapipe()
arcface_model = load_arcface_model()
# ----------------------------------------------------

# ----------------- HELPERS -----------------
def load_np_image_from_upload(upload) -> np.ndarray:
    return np.array(Image.open(upload).convert("RGB"))

def load_np_image_from_url(url: str) -> np.ndarray:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return np.array(Image.open(BytesIO(r.content)).convert("RGB"))

def detect_and_crop_retina(img_rgb: np.ndarray) -> List[np.ndarray]:
    """
    Returns list of aligned face crops (RGB uint8) using RetinaFace via DeepFace.extract_faces.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True
        )
    except Exception:
        return []
    outs = []
    for f in faces:
        face = f["face"]
        # DeepFace returns float32 in range [0,1]; convert to uint8 RGB
        if face.dtype != np.uint8:
            face = (face * 255).clip(0, 255).astype(np.uint8)
        outs.append(face)
    return outs

def count_faces_mediapipe(img_rgb: np.ndarray) -> int:
    res = face_detector_mp.process(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return len(res.detections) if res.detections else 0

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def embed_face(face_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Embed a single cropped face (RGB) with ArcFace model.
    """
    try:
        rep = DeepFace.represent(
            img_path=face_rgb,
            model_name=EMBED_MODEL,
            detector_backend="skip",
            enforce_detection=False,
            model=arcface_model
        )
        if rep and isinstance(rep, list):
            return np.array(rep[0]["embedding"], dtype=np.float32)
    except Exception:
        return None
    return None

def robust_match(ref_img: np.ndarray, new_img: np.ndarray) -> Tuple[int, Optional[float]]:
    """
    Detect/crop both with RetinaFace, embed with ArcFace, cosine match.
    If anything fails -> fallback to DeepFace.verify (slow but sure).
    Returns (match_flag, cosine_distance or None)
    """
    ref_faces = detect_and_crop_retina(ref_img)
    new_faces = detect_and_crop_retina(new_img)

    if len(ref_faces) == 0 or len(new_faces) == 0:
        # fallback verify directly on full frames
        try:
            vr = DeepFace.verify(ref_img, new_img,
                                 model_name=EMBED_MODEL,
                                 detector_backend=DETECTOR_BACKEND,
                                 enforce_detection=False)
            return (1 if vr.get("verified", False) else 0,
                    vr.get("distance", None))
        except Exception:
            return 0, None

    # Use the first (since you guarantee one face in ref)
    ref_emb = embed_face(ref_faces[0])
    new_emb = embed_face(new_faces[0])

    if ref_emb is None or new_emb is None:
        # Fallback verify
        try:
            vr = DeepFace.verify(ref_img, new_img,
                                 model_name=EMBED_MODEL,
                                 detector_backend=DETECTOR_BACKEND,
                                 enforce_detection=False)
            return (1 if vr.get("verified", False) else 0,
                    vr.get("distance", None))
        except Exception:
            return 0, None

    dist = cosine_distance(ref_emb, new_emb)
    return (1 if dist < COSINE_THRESH else 0, float(dist))

def get_iris_gaze_flag(img_rgb: np.ndarray, tol: float = GAZE_TOL) -> int:
    res = face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not res.multi_face_landmarks:
        return 0
    lm = res.multi_face_landmarks[0].landmark

    left_iris_idx  = [473, 474, 475, 476]
    right_iris_idx = [468, 469, 470, 471]
    left_eye_idx   = [33, 133, 159, 145]
    right_eye_idx  = [263, 362, 386, 374]

    try:
        def center(idxs):
            pts = np.array([[lm[i].x, lm[i].y] for i in idxs])
            return pts.mean(axis=0)

        def box(idxs):
            pts = np.array([[lm[i].x, lm[i].y] for i in idxs])
            return pts.min(0), pts.max(0)

        li_c = center(left_iris_idx)
        ri_c = center(right_iris_idx)
        le_min, le_max = box(left_eye_idx)
        re_min, re_max = box(right_eye_idx)

        def offset(c, mn, mx):
            center_xy = (mn + mx) / 2
            rng = (mx - mn)
            rng[rng == 0] = 1e-6
            return np.abs((c - center_xy) / rng)

        lo = offset(li_c, le_min, le_max)
        ro = offset(ri_c, re_min, re_max)
        return 1 if (lo < tol).all() and (ro < tol).all() else 0
    except IndexError:
        return 0

def head_pose_gaze_flag(img_rgb: np.ndarray,
                        yaw_max: float = YAW_MAX_DEG,
                        pitch_max: float = PITCH_MAX_DEG) -> int:
    res = face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not res.multi_face_landmarks:
        return 0
    lm = res.multi_face_landmarks[0].landmark
    h, w = img_rgb.shape[:2]
    try:
        image_points = np.array([
            (lm[1].x*w,   lm[1].y*h),   # Nose tip
            (lm[152].x*w, lm[152].y*h), # Chin
            (lm[33].x*w,  lm[33].y*h),  # Left eye left corner
            (lm[263].x*w, lm[263].y*h), # Right eye right corner
            (lm[61].x*w,  lm[61].y*h),  # Left mouth corner
            (lm[291].x*w, lm[291].y*h)  # Right mouth corner
        ], dtype="double")
    except IndexError:
        return 0

    model_points = np.array([
        (0.0,    0.0,    0.0),     # Nose tip
        (0.0,  -63.6,  -12.5),     # Chin
        (-43.3, 32.7,  -26.0),     # Left eye left corner
        (43.3,  32.7,  -26.0),     # Right eye right corner
        (-28.9,-28.9,  -24.1),     # Left mouth corner
        (28.9, -28.9,  -24.1)      # Right mouth corner
    ], dtype="double")

    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    try:
        success, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix,
                                        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return 0
    except cv2.error:
        return 0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
    yaw   = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

    return 1 if abs(yaw) < yaw_max and abs(pitch) < pitch_max else 0

def is_looking_at_screen(img_rgb: np.ndarray) -> int:
    iris_flag = get_iris_gaze_flag(img_rgb)
    if iris_flag == 1:
        return 1
    return head_pose_gaze_flag(img_rgb)
# -----------------------------------------

# ----------------- UI -----------------
st.title("📷 Interview Frame Validator (2-frame check)")
st.write("Frame 1 is trusted. We compare Frame 2 to it. If iris gaze fails, head‑pose is used.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Frame 1 (Reference)")
    up1 = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"], key="ref_up")
    url1 = st.text_input("...or URL for Reference Image", key="ref_url")

with c2:
    st.subheader("Frame 2 (Compare)")
    up2 = st.file_uploader("Upload New Image", type=["jpg", "jpeg", "png"], key="new_up")
    url2 = st.text_input("...or URL for New Image", key="new_url")

def fetch_image(upload, urlfield):
    if upload is not None:
        return load_np_image_from_upload(upload)
    if urlfield:
        try:
            return load_np_image_from_url(urlfield)
        except Exception as e:
            st.error(f"URL error: {e}")
    return None

ref_img = fetch_image(up1, url1)
new_img = fetch_image(up2, url2)

if ref_img is not None:
    st.image(ref_img, caption="Reference Frame", use_column_width=True)
if new_img is not None:
    st.image(new_img, caption="New Frame", use_column_width=True)

if ref_img is None or new_img is None:
    st.info("Upload both images to proceed.")
    st.stop()
# --------------------------------------

# ----------------- PROCESSING -----------------
match_flag, dist = robust_match(ref_img, new_img)
num_faces_new = count_faces_mediapipe(new_img)
look_flag = is_looking_at_screen(new_img)

violations = int(not (match_flag == 1 and num_faces_new == 1 and look_flag == 1))

result: Dict[str, object] = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "matching_faces": match_flag,
    "cosine_distance": None if dist is None else round(float(dist), 4),
    "num_faces_new_frame": num_faces_new,
    "looking_at_screen_new_frame": look_flag,
    "violations_detected": violations
}

st.subheader("📊 Result")
st.json(result)

with st.expander("Debug (MP face boxes on New Frame)"):
    res = face_detector_mp.process(cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
    dbg = new_img.copy()
    if res.detections:
        for det in res.detections:
            rbb = det.location_data.relative_bounding_box
            h, w = dbg.shape[:2]
            x, y = int(rbb.xmin * w), int(rbb.ymin * h)
            bw, bh = int(rbb.width * w), int(rbb.height * h)
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    st.image(dbg, caption="New Frame with MP Face Box", use_column_width=True)

if st.button("Clear all and start over"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()
