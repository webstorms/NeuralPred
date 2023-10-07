import os


root = "/home/luketaylor/PycharmProjects/NeuralPred"


def build_dataset(dataset, ntau, nlat, nspan, scale, model, layer, n_pca):
    os.system(f"python build_pca.py --root={root} --dataset={dataset} --ntau={ntau} --nlat={nlat} --nspan={nspan} --scale={scale} --model={model} --layer={layer} --n_pca={n_pca}")


for dataset in ["cadena", "multi_pvc1"]:
    for layer in [1, 2, 3, 4]:
        for scale in [0.66, 1.0, 1.5]:
                build_dataset(dataset, ntau=25, nlat=1, nspan=3, scale=scale, model=f"stack_prediction_4-6-2_True", layer=layer, n_pca=500)