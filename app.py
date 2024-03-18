import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import easyocr

def compare_bbox(result):

    # Load the image
    image = result.orig_img
    image2 = image.copy()

    _dict = result.names
    boxes = result.boxes

    check = 0

    for box in boxes:
        if _dict[int(box.cls.cpu().item())]=="book":
            check=1
            box_interest = box
            break

    for box in boxes:
        if _dict[int(box.cls.cpu().item())]=="person":
            x1, y1, x3, y3 = [int(i) for i in box.xyxy.cpu().data.tolist()[0]]
            
            cv2.rectangle(image2, (x1, y1), (x3, y3), color=(0,0,255), thickness=5)

            break

    if check==0:
        cv2.putText(image2, "Couldn't Detect ID", org=(300, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 4, color = (255, 0, 0), thickness=5)
        
        return image2, 0

    # Define the coordinates of the bounding box
    x1, y1, x3, y3 = [int(i) for i in box_interest.xyxy.cpu().data.tolist()[0]]
    # print('Coord', x1, y1, x3, y3)

    cv2.rectangle(image2, (x1, y1), (x3, y3), color=(255,0,0), thickness=5)

    return image2, box_interest


st.title('ID Document Detector')

# Display image
st.subheader('Upload an Image')
st.write('Only .tif allowed')
img_file = st.file_uploader(label="Upload your picture", type=['tiff'])

col1, col2 = st.columns(2)

infer_model = YOLO("yolov8s.pt")
 
flag = False

if img_file!= None:
    with col1:
        st.subheader("Your Image")
        st.image(img_file, width=300)

    with col2:
        st.subheader("Objects detected")
        
        image = Image.open(img_file)
        img_array = np.array(image)
        result = infer_model(img_array, stream=False)[0] 

        img2, box_interest = compare_bbox(result)

        if box_interest!=0:
            st.image(img2, width=300)
            flag = True

            x1, y1, x3, y3 = [int(i) for i in box_interest.xyxy.cpu().data.tolist()[0]]
            sub_img = img_array[y1:y3, x1:x3, :3]
        else:
            st.write("Sorry we couldn't detect any object in this image")


if flag:
    st.divider()
    st.subheader('OCR Reading')

    reader = easyocr.Reader(['en'])
    fields = reader.readtext(sub_img)

    for field in fields:
        # print(t)
        bbox, text, score = field

        try:
            cv2.rectangle(sub_img, bbox[0], bbox[2], (0, 255, 0), 5)
        except:
            pass

    st.image(sub_img)
        
    



    
