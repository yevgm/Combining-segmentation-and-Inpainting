# Combining-segmentation-and-Inpainting

# Environment setup

Clone the repo:
`git clone https://github.com/saic-mdal/lama.git`

Conda
    
  ```
  % Install conda for Linux, for other OS download miniconda at https://docs.conda.io/en/latest/miniconda.html
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
  $HOME/miniconda/bin/conda init bash

  cd lama
  conda env create -f conda_env.yml
  conda activate lama
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
  pip install pytorch-lightning==1.2.9
  ```
  
  # Train and Eval
