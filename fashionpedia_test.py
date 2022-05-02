import json
import os
from PIL import Image, ImageDraw, ImageChops
import numpy
import pycocotools.mask as maskUtils


CLOTHES_CATEGORIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19, 21, 22, 23, 24, 25]
AREA_THRESHOLD = 3000

dataset_path = "/home/salvatori/datasets/FashionPedia/"
json_file_path = os.path.join(dataset_path, "instances_attributes_val2020.json")

with open(json_file_path, "r") as j:
    data = json.loads(j.read())
    img_ids_to_metadata = {m["id"]: m for m in data["images"]}
    idx2category = {i['id']: i['name'] for i in data['categories']}
    for annotation in data["annotations"]:
        if not annotation["category_id"] in CLOTHES_CATEGORIES or annotation["area"] < AREA_THRESHOLD or annotation["image_id"]!=17039:
            continue
        print(annotation["image_id"])
        print(idx2category[annotation["category_id"]])
        img_file = img_ids_to_metadata[annotation["image_id"]]["file_name"]
        img_path = os.path.join(dataset_path, "test", img_file)
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            if isinstance(annotation["segmentation"], dict): # RLE annotation
                maskIm = Image.fromarray(maskUtils.decode(annotation["segmentation"]))
            else:
                maskIm = Image.new("1", image.size, 1)
                # Select the clothes using its segmentation
                imgDraw = ImageDraw.Draw(maskIm)
                for polygon in annotation["segmentation"]:
                    imgDraw.polygon(polygon, fill=0)
            diff_img = ImageChops.add(image, maskIm.convert("RGB"))
            x, y, w, h = tuple(annotation["bbox"])
            diff_img = diff_img.crop((x, y, x + w, y + h))
            diff_img.save("segmented.png")
            image.save("orig.png")
