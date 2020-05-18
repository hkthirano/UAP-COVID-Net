# UAP-COVID-Net

ディレクトリ

```
.
|-- COVID-Net
|   |-- data
|   |   |-- train
|   |   `-- test
|   |-- model
|   |   |-- COVIDNet-CXR-Small
|   |   `-- COVIDNet-CXR-Large
|   `-- inference.py
|-- UAP-COVID-Net
|   `-- generate_noise.py
|-- covid-chestxray-dataset
|-- Figure1-COVID-chestxray-dataset
`-- ctualmed-COVID-chestxray-dataset
```

1. オリジナルの[lindawangg/COVID-Net](https://github.com/lindawangg/COVID-Net)の`inference.py`が動くようにする
    - Requirements to install on your system
    - How to generate COVIDx dataset
      - データセット
        - `covid-chestxray-dataset`
        - `Figure1-COVID-chestxray-dataset`
        - `rsna-pneumonia-detection-challenge`
      - スクリプト
        - `create_COVIDx_v2.ipynb`
    - Download a model from the pretrained models section
      - モデル
        - `COVIDNet-CXR Small`
        - `COVIDNet-CXR Large`

2. Install the UAP method and Keras.
  - `pip install git+https://github.com/hkthirano/adversarial-robustness-toolbox`
  - `pip install keras` : データの前処理に`to_categorical`を使いたいだけ

3. Generate a UAP

```
# $ pwd
# > UAP-COVID-Net

python generate_nontargeted_uap.py
python generate_targeted_uap.py --target COVID-19
python generate_random_uap.py
```