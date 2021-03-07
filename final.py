import utils
from functions import *

content_image = load_image(r'C:\Users\Admin\Desktop\worli_nst.jpg')
style_image = load_image(r"C:\Users\Admin\Desktop\style_nst3.jpg")
content_layer_ids = [4]
style_layer_ids = list(range(13))
f = style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0)
