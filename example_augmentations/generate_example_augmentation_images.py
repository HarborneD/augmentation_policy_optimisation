import sys
import os

import cv2

import numpy as np


try:
    from PIL import Image
except:
    from PILLOW import Image

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from evaluator.augmentation_transforms import TRANSFORM_NAMES, NAME_TO_TRANSFORM, pil_wrap, pil_unwrap
from evaluator.data_utils import unpickle

def DisplayImage(img):
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def SaveImage(image, output_path, flip_channels = False):
    output_image = image
    if(flip_channels):
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    if(output_image.max() <= 1):
        output_image = output_image*255
    im = Image.fromarray(output_image.astype(np.uint8))
    im.save(output_path)

print(TRANSFORM_NAMES)

transforms_to_create_examples_of = ['FlipLR', 'FlipUD', 'Invert', 'Rotate', 'Contrast', 'Brightness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY']

output_dir = "augmentation_examples"

data_path = "/media/harborned/ShutUpN/datasets/cifar/cifar-10-batches-py"

image_batch_path = os.path.join(data_path,"data_batch_1")

images = unpickle(image_batch_path)
all_data = images["data"].reshape(10000, 3072)
all_data = all_data.reshape(-1, 3, 32, 32)
all_data = all_data.transpose(0, 2, 3, 1).copy()
all_data = np.float32(all_data / 255.0)
# for i in range(10):
#     DisplayImage(all_data[i])


# DisplayImage(all_data[7])

original_image = all_data[7]
output_path = os.path.join(output_dir,"original.jpg")
SaveImage(original_image, output_path)

for transform_name in transforms_to_create_examples_of:
    print(transform_name+"& \\includegraphics[width=0.1\\textwidth, height=0.1\\textwidth]{augmentation_examples/"+transform_name+".jpg} \\\\")
    output_path = os.path.join(output_dir,transform_name+".jpg")

    xform_fn = NAME_TO_TRANSFORM[transform_name].pil_transformer(1.0, 0.8)
    augmented_image = pil_unwrap(xform_fn(pil_wrap(np.copy(original_image))))

    # DisplayImage(augmented_image)
    SaveImage(augmented_image, output_path) 