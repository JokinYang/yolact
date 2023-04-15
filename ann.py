import copy

import albumentations as A
import os.path
from pathlib import Path
import random
import shutil
import json
import cv2 as cv
from tqdm import tqdm

random_scale = A.Compose([A.RandomScale(scale_limit=(-0.2, 0.2), p=1), ], keypoint_params=A.KeypointParams(format='xy'),
                         additional_targets={'name': 's'})

horizontal_flap = A.Compose([A.HorizontalFlip(p=1), ], keypoint_params=A.KeypointParams(format='xy'),
                            additional_targets={'name': 'f'})

pixel_dropout = A.Compose([A.PixelDropout(p=1), ], keypoint_params=A.KeypointParams(format='xy'),
                          additional_targets={'name': 'd'})

gauss_noise = A.Compose([A.GaussNoise(p=1), ], keypoint_params=A.KeypointParams(format='xy'),
                        additional_targets={'name': 'g'})

# random_rotate = A.Compose([A.Rotate(limit=10, p=1), ], keypoint_params=A.KeypointParams(format='xy'),
#                           additional_targets={'name': 'r'})
affine = A.Compose([A.Affine(scale=(0.8, 1.2), keep_ratio=True, translate_percent=0.05, shear=10, p=1), ],
                   keypoint_params=A.KeypointParams(format='xy'),
                   additional_targets={'name': 'a'})

FUNCTION_LIST = [
    random_scale, horizontal_flap, pixel_dropout, gauss_noise,
    # affine
]

output_dir = Path(r"E:\Workspace\PycharmProjects\yolact-pytorch\datasets\ann")

input_dir = Path(r"E:\Workspace\PycharmProjects\yolact-pytorch\datasets\before")

for j in tqdm(list(input_dir.glob("*.json"))):
    i_path = j.parent / j.name.replace(".json", ".jpg")

    o = json.loads(j.read_text())
    points = o['shapes'][0]['points']
    img = cv.imread(str(i_path))

    for F in FUNCTION_LIST:
        ret = F(image=img, keypoints=points)
        ret_img = ret['image']
        ret_points = ret['keypoints']
        F_name = F.additional_targets.get('name', ' ')

        output_name = j.name.rsplit('.')[0] + f"_{F_name}_"

        d = copy.deepcopy(o)
        d['imageData'] = None
        d['imagePath'] = output_name + ".jpg"
        d['shapes'][0]['points'] = list(map(lambda x: list(map(float, x)), ret_points))

        cv.imwrite(str(output_dir / (output_name + ".jpg")), ret_img)
        try:
            (output_dir / (output_name + ".json")).write_text(json.dumps(d, indent=2))
        except Exception as e:
            print(d)
            raise e
