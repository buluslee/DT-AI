import os
import xml.etree.ElementTree as ET
from pathlib import Path

dir_path = 'D:/faster-rcnn-pytorch-master/VOCdevkit/VOC2007/Annotations'

listdir = os.listdir(dir_path)
for file in listdir:
    if file.endswith('.xml'):
        file_path = os.path.join(dir_path, file)
        tree = ET.parse(file_path)
        root = tree.getroot()
        image_path = root.find("path").text
        voc_file_name = Path(file).stem
        image_file_name, image_ext = os.path.splitext(image_path)
        new_image_path = os.path.join(dir_path, voc_file_name + image_ext)
        os.rename(image_path, new_image_path)
        print(image_file_name)
        print(voc_file_name)

