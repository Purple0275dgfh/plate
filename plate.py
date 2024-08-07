import streamlit as st
import cv2
import pytesseract
import numpy as np
import tempfile
from PIL import Image

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Streamlit app
def main():
    st.title("Number Plate Recognition")

    # Upload video file
    uploaded_file = st.file_uploader("Choose a video...", type="mp4")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Read the video file
        video = cv2.VideoCapture(temp_file_path)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
            plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                plate = frame[y:y + h, x:x + w]

                # Convert the plate image to the PIL format
                pil_plate = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))

                # Use Tesseract to do OCR on the plate image
                text = pytesseract.image_to_string(pil_plate)
                st.write(f"Detected Plate: {text}")

            # Display frame
            st.image(frame, channels="BGR")

        video.release()

if __name__ == "__main__":
    main()
