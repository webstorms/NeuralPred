Part of the accompanying code repository to the "Hierarchical temporal prediction captures motion processing along the visual pathway" paper. This repo contains the code to perform the neural V1 fits.

## Installation
```bash
conda env create -f envs/environment.yml
conda env create -f envs/prednet.yml
```

## Clone control model repos
Clone repos into the project's ```dependencies``` folder.
```bash
git clone https://github.com/coxlab/prednet.git
git clone https://github.com/sacadena/Cadena2019PlosCB.git
git clone  https://github.com/ben-willmore/bwt.git
git clone  https://github.com/webstorms/StackTP.git
```

## Download V1 data
Image V1 dataset obtainable from Cadena et al. 2019: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006897
Movie V1 dataset obtainable from Nahaus and Ringach, 2007: https://crcns.org/data-sets/vc/pvc-1

Movie dataset needs to be further processed using ```scripts/build_dataset.py``` and both tensorized responses are built using ```scripts/neuron_to_tensor.py```.

## Building PC model activity dataset
```bash
conda activate neuralpred
```
All build scripts can be found under ```scripts/build_pca```. You will need to activate ```prednetbuild``` for building the prednet responses.

## Training
All readout fitting scripts can be found under ```scripts/fit_models```.

## Inspecting models
See ```notebooks/Inspection.ipynb``` to view results.

## License
Code released under the MIT license.