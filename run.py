from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from skimage.transform import resize

import skimage.io
import numpy as np


def read_img(path):
    img = skimage.img_as_float(skimage.io.imread(path)).astype(np.float32)
    print("Original Image Shape: ", img.shape)
    return img


def preproc(img, h, w):
    img_r = skimage.transform.resize(img, (h, w))
    img_r = img_r.swapaxes(1, 2).swapaxes(0, 1)
    img_r = img_r[(2, 1, 0), :, :]
    img_r = img_r * 255 - 128
    img_r = img_r[np.newaxis, :, :, :].astype(np.float32)
    return img_r


def read_model(init_net, pred_net):
    with open(init_net, "rb") as f:
        init_net = f.read()

    with open(pred_net, "rb") as f:
        predict_net = f.read()

    p = workspace.Predictor(init_net, predict_net)
    return p


if __name__ == "__main__":

    gender_model = read_model("models/gender/init_net.pb",
                              "models/gender/predict_net.pb")
    age_model = read_model("models/lap/init_net.pb",
                           "models/lap/predict_net.pb")

    while True:

        img_path = input(f"\n{'-' * 120}\n"
                         "Input image path or URL\n").strip(r"'")

        img = preproc(read_img(img_path), 224, 224)

        # gender prediction
        gender_results = gender_model.run({'data': img})[0][0]
        gen = "woman" if np.argmax(gender_results) == 0 else "man"
        print(f"Estimated gender: {gen} (p={np.max(gender_results)})")

        # age prediction
        age_results = age_model.run({'data': img})[0][0]

        est = np.sum([i * p
                      for i, p in enumerate(age_results)])
        print(f"Estimated age: {est}")


