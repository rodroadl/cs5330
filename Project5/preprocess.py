import os
from wand.color import Color
from wand.image import Image
from wand.display import display
dp = "data/jgreek/"

for fp in os.listdir(dp):
    with Image(filename=dp+fp) as img:
        with img.clone() as i:
            i.rotate(90)
            i.transform_colorspace('gray')
            i.white_threshold("#555")
            i.resize(133,133)
            i.save(filename=dp+"0"+fp)