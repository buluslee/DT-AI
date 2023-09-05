import time

import pandas as pd
import streamlit as st
from predict import main
from PIL import Image

st.write("这是我写的一个小项目，花的分类,目前只支持5类类别：菊花，蒲公英，玫瑰，向日葵，郁金香")
st.subheader("欢迎使用TJL写的小项目，花的分类")
label = {"daisy": "菊花", "dandelion": "蒲公英", "roses": "玫瑰", "sunflowers": "向日葵", "tulips": "郁金香"}
upload_file = st.file_uploader(label="上传一张花的图片,支持jpg,png,jpeg")
if upload_file is not None:
    img = Image.open(upload_file)
    st.success("上传成功")
    time.sleep(0.5)
    st.write("显示图片")
    st.image(img)
else:
    st.stop()  # 退出

if st.button("显示结果"):
    with st.spinner("模型识别中......"):
        l = main(img)
    st.success("类别为:  {}".format(l[0]))
