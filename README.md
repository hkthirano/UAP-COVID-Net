# UAP-COVID-Net

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

1. Check [lindawangg/COVID-Net : Table of Contents](https://github.com/lindawangg/COVID-Net#table-of-contents) for installation instructions.
- Requirements to install on your system
- How to generate COVIDx dataset
  - Download the datasets listed above
    - `covid-chestxray-dataset`
    - `Figure1-COVID-chestxray-dataset`
    - `rsna-pneumonia-detection-challenge`
  - Use `create_COVIDx_v2.ipynb`
- Download a model from the pretrained models section
  - `COVIDNet-CXR Small`
  - `COVIDNet-CXR Large`

2. Install the UAP method and Keras (to generate dataset).
- `pip install git+https://github.com/hkthirano/adversarial-robustness-toolbox`
- `pip install keras`

3. Generate a UAP.

```
# $ pwd
# > UAP-COVID-Net

python generate_nontargeted_uap.py
python generate_targeted_uap.py --target COVID-19
python generate_random_uap.py
```

4. Results
