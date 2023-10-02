import cv2
import numpy as np
import streamlit as st
import os

st.title(":green[WELCOME TO OBJECT IDENTIFIER]")
st.header(":blue[THIS SITE WILL IDENTIFY OBJECTS BASED ON YOUR GIVEN IMAGE]")
st.caption("__*LIGHTS ARE YET TO BE TURNED ON*__")

# Create a file uploader widget
uploaded_file = st.file_uploader(":orange[UPLOAD IMAGE FOR CLASSIFICATION IDENTIFICATION]")

# Initialize temp_image_path with None
temp_image_path = None

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.read())

    img = cv2.imread(temp_image_path)  # object reader

    all_rows = open("E:/ocv/model/synset_words.txt").read().strip().split("\n")
    classes = [r[r.find(' ')+ 1:] for r in all_rows]

    if img is not None:
        net = cv2.dnn.readNetFromCaffe("E:/ocv/model/bvlc_googlenet.prototxt", "E:/ocv/model/bvlc_googlenet.caffemodel")
        blob = cv2.dnn.blobFromImage(img, 1, (224, 224))
        net.setInput(blob)
        outp = net.forward()
        idx = np.argsort(outp[0])[::-1][:5]

        b = st.button("GIVE MOST PROBABLE OBJECTS WITH THIER CORRESPONDING PROBABILITY")
        if b:
            st.balloons()
            for (i, id) in enumerate(idx):
                st.write('{}.:violet[{}]: :green[Probability {:.3f}%]'.format(i+1, classes[id], outp[0][id]*100))
        else:
            pass
    else:
        st.warning("Failed to read the uploaded image.")
else:
    st.warning("NO IMAGE HAS BEEN GIVEN")

# Clean up: Delete the temporary image file if it exists
if temp_image_path is not None and os.path.exists(temp_image_path):
    os.remove(temp_image_path)
