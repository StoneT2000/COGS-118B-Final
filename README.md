# COGS-118B-Final

## Setup

Run the following to setup conda and get a local install
```
conda env create -f environment.yml
conda activate cogs118b
pip install -e .
```

To get the tiny image dataset, download and unzip 

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip
```

Run the script in notebooks/stone/data.ipynb to then process the data
