conda create -y -n TeethSeg python=3.10
conda activate TeethSeg
yes | pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
yes | pip install -r requirements.txt
yes | pip install ipykernel
python -m ipykernel install --user --name TeethSeg
