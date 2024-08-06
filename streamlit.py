import streamlit as st
import cv2
import numpy as np
import imutils
import easyocr
from PIL import Image

st.title("Automatic Number Plate Recognition")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB), caption='Grayscale Image', use_column_width=True)
    
    # Noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    st.image(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB), caption='Bilateral Filtered Image', use_column_width=True)
    
    # Edge detection
    edged = cv2.Canny(bfilter, 30, 200)
    st.image(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB), caption='Edge Detected Image', use_column_width=True)
    
    # Finding contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    if location is not None:
        # Create mask and extract license plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        st.image(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB), caption='Masked Image', use_column_width=True)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
        st.image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), caption='Cropped Image', use_column_width=True)

        # Text recognition
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        
        if result:
            text = result[0][-2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
            
            st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption='Processed Image', use_column_width=True)
            st.write(f"Detected Text: {text}")
        else:
            st.write("No text found")
    else:
        st.write("License plate contour not found")
else:
    st.write("Please upload an image.")
