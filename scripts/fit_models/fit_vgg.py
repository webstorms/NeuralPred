import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

root = "/home/luke/PycharmProjects/NeuralPred"


def build_dataset(dataset, ntau, nlat, nspan, scale, model, layer, n_pca):
    os.system(f"python fit.py --root={root} --dataset={dataset} --ntau={ntau} --nlat={nlat} --nspan={nspan} --scale={scale} --model={model} --layer={layer} --n_pca={n_pca}")


for dataset in ["multi_pvc1"]:
    for scale in [0.66, 1.0, 1.5]:
        build_dataset(dataset, ntau=25, nlat=1, nspan=3, scale=scale, model="vgg", layer="3.1", n_pca=500)


for dataset in ["cadena"]:
    for scale in [0.66, 1.0, 1.5]:
        build_dataset(dataset, ntau=25, nlat=0, nspan=1, scale=scale, model="vgg", layer="3.1", n_pca=500)
