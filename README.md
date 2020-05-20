# UAP-COVID-Net
This repository contains the dataset and codes for reproducing the results in [paper title](url).

## Usage

```
# Directories
.
├── COVID-Net
│   ├── create_COVIDx_v2.ipynb
│   ├── data
│   │   ├── train
│   │   └── test
│   ├── inference.py
│   └── models
│       └── COVIDNet-CXR-Small
├── UAP-COVID-Net
│   ├── generate_nontargeted_uap.py
│   ├── generate_random_uap.py
│   ├── generate_targeted_uap.py
│   ├── output
│   └── uap_utils.py
├── covid-chestxray-dataset
├── Figure1-COVID-chestxray-dataset
└── rsna-pneumonia-detection-challenge
```

1. See [lindawangg/COVID-Net: Table of Contents](https://github.com/lindawangg/COVID-Net#table-of-contents) for installation.
- Check the requirements
- Generate the COVIDx dataset
  - Download the following datasets
    - `covid-chestxray-dataset`
    - `Figure1-COVID-chestxray-dataset`
    - `rsna-pneumonia-detection-challenge`
  - Use `create_COVIDx_v2.ipynb`
- Download the COVID-Net models available [here](https://github.com/lindawangg/COVID-Net/blob/master/docs/models.md)
  - `COVIDNet-CXR Small`
  - `COVIDNet-CXR Large`

2. Install the UAP methods and Keras.
- `pip install git+https://github.com/hkthirano/adversarial-robustness-toolbox`
- `pip install keras`

3. Generate a UAP.

```
# $ pwd
# > UAP-COVID-Net

# non-targeted UAP
python generate_nontargeted_uap.py

# UAP for targeted attacks to COVID-19
python generate_targeted_uap.py --target COVID-19 
# `target` argument indicates the target class: normal, pneumonia, or COVID-19 (default).

# random UAP
python generate_random_uap.py
```

4. Results
