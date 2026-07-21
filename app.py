import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Page config
st.set_page_config(page_title="Digit Recognizer", page_icon="🧠", layout="centered")

# 🎨 CUSTOM UI (YOUR BACKGROUND IMAGE)
st.markdown("""
<style>

/* BACKGROUND IMAGE */
.stApp {
    background-image: url("background.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* GLASS EFFECT BOX */
.box {
    background: rgba(0, 0, 0, 0.6);
    color: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}

/* TEXT STYLING */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    text-shadow: 2px 2px 12px rgba(0,0,0,0.9);
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #e0e0e0;
}
.team {
    text-align: center;
    font-size: 18px;
    color: #ffd54f;
    margin-bottom: 25px;
    font-style: italic;
}
.result {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    color: #00e676;
}

/* BUTTON STYLE */
button[kind="primary"] {
    width: 100%;
    border-radius: 10px;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("digit_model.pkl")

# 🧠 HEADER
st.markdown('<div class="title">🧠 Digit Recognition App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw or Upload a Digit and Let AI Predict It</div>', unsafe_allow_html=True)
st.markdown('<div class="team">Developed by Team SWAT 🚀</div>', unsafe_allow_html=True)

st.info("👉 Draw a LARGE, CENTERED digit with clear strokes")

# 🎨 CANVAS
st.markdown('<div class="box">', unsafe_allow_html=True)

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("🧹 Clear"):
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# 📤 UPLOAD
st.markdown('<div class="box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# 🔍 PREDICT BUTTON (BOTTOM)
predict_btn = st.button("🚀 Predict")

# 🔥 IMAGE PROCESSING (UNCHANGED)
def process_image(image, is_upload=False):
    image = image.convert("L")

    if is_upload:
        image = ImageOps.autocontrast(image)
        img = np.array(image)

        img = 255 - img
        img = np.where(img > 30, img, 0)

        coords = np.column_stack(np.where(img > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            img = img[y_min:y_max+1, x_min:x_max+1]

        img = Image.fromarray(img.astype("uint8")).resize((20, 20))

        new_img = np.zeros((28, 28))
        new_img[4:24, 4:24] = np.array(img)

        img = new_img / 255.0

    else:
        img = np.array(image)

        img = 255 - img
        img = (img > 50) * 255

        coords = np.column_stack(np.where(img > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            img = img[y_min:y_max+1, x_min:x_max+1]

        img = Image.fromarray(img.astype("uint8")).resize((20, 20))

        new_img = np.zeros((28, 28))
        new_img[4:24, 4:24] = np.array(img)

        img = new_img / 255.0

    return img, img.flatten().reshape(1, -1)


# 🔍 PREDICTION
if predict_btn:

    canvas_drawn = False

    if canvas_result.image_data is not None:
        img_arr = canvas_result.image_data.astype("uint8")
        gray = np.mean(img_arr, axis=2)

        if np.mean(gray) < 250:
            canvas_drawn = True

    # ❌ BOTH USED
    if canvas_drawn and uploaded_file is not None:
        st.error("❌ Please use only ONE input: either draw OR upload")

    # ✏ DRAW
    elif canvas_drawn:
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")

        processed_img, data = process_image(img, False)

        st.image(processed_img, caption="Processed Image", clamp=True)

        probs = model.predict_proba(data)
        prediction = np.argmax(probs)
        confidence = np.max(probs)

        st.markdown(f'<div class="result">Prediction: {prediction}</div>', unsafe_allow_html=True)
        st.write(f"Confidence: {confidence:.2f}")

    # 📤 UPLOAD
    elif uploaded_file is not None:
        img = Image.open(uploaded_file)

        processed_img, data = process_image(img, True)

        st.image(processed_img, caption="Processed Image", clamp=True)

        probs = model.predict_proba(data)
        prediction = np.argmax(probs)
        confidence = np.max(probs)

        st.markdown(f'<div class="result">Prediction: {prediction}</div>', unsafe_allow_html=True)
        st.write(f"Confidence: {confidence:.2f}")

    # ⚠ NOTHING
    else:
        st.warning("⚠ Please draw or upload an image")

