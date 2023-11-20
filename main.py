from style import style_transfer
from PIL import Image, ImageEnhance
import streamlit as st 
import numpy as np 
import os 

def main():
    # 设置Streamlit应用程序的标题
    st.set_page_config(page_title="styleTransfer", layout="wide")
    # 设置 sidebar 的标题
    st.sidebar.title("风格转移")

    # 让用户上传文件并显示
    uploaded_file = st.sidebar.file_uploader("请选择一张图片文件", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption='原始图像.', use_column_width=True)
        
        # 图片处理
        enhance_type = st.sidebar.radio("请选择要进行的图片处理类型", ("candy", "feathers", "la_muse", "mosaic","the_scream","the_wave","udnie"))
        if enhance_type == "candy":
            weights = "candy.t7"
            image = style_transfer(weights, np.asarray(image))
        elif enhance_type == "feathers":
            weights = "feathers.t7"
            image = style_transfer(weights, np.asarray(image))
        elif enhance_type == "la_muse":
            weights = "la_muse.t7"
            image = style_transfer(weights, np.asarray(image))
        elif enhance_type == "mosaic":
            weights = "mosaic.t7"
            image = style_transfer(weights, np.asarray(image))
        elif enhance_type == "the_scream":
            weights = "the_scream.t7"
            image = style_transfer(weights, np.asarray(image))
        elif enhance_type == "the_wave":
            weights = "the_wave.t7"
            image = style_transfer(weights, np.asarray(image))
        elif enhance_type == "udnie":
            weights = "udnie.t7"
            image = style_transfer(weights, np.asarray(image))
        else:
            pass
        image = Image.fromarray(image.astype(np.uint8))
        # 显示处理后的图片
        st.image(image, caption='Enhanced Image.', use_column_width=True)
if __name__ == '__main__':
    #os.system('apt-get -y update')
    #os.system('apt-get -y install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev freeglut3-dev')
    main()
