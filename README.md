# TeethSeg
## Install
```shell
conda create -n TeethSeg python=3.10
conda activate TeethSeg
```

```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
```shell
pip install -r requirements.txt
```
```shell
pip install ipykernel
python -m ipykernel install --user --name TeethSeg
```

## Folder
```shell
|- dataset
    |- teeth
        |- data50
            |- image
            |- label
            |- center
            |- data.json (use exp.ipynb -> Preprocess cell)
            |- data_center.json
|- TeethSeg
```
