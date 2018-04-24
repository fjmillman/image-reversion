# image-reversion
Based on [pix2pix](https://github.com/phillipi/pix2pix) by Isola et al.

## Setup
### Pre-requisites
- Tensorflow 1.4.1

### Recommended
- Linux with Tensorflow GPU edition + cuDNN

### Getting Started
```bash
# Clone this repository:
git clone git@github.com:fjmillman/image-reversion.git
 
# Change directory into the repository
cd image-reversion
 
# Download the datasets
bash download-datasets.sh
 
# Train the model
python main.py --input_dir datasets/{dataset}/train
    --output_dir datasets/{dataset} --mode train
 
# Test the model
python main.py --input_dir datasets/{dataset}/test
    --output_dir datasets/{dataset}/results --mode test
    --checkpoint datasets/{dataset}
```

## Testing the outputs
The outputs of testing can be compared to see whether the model has successfully managed to revert the image enhancements. Two different operations can be run to find the image similarity metric averages of the dataset or to get a sum of absolute difference heatmap between each pair of images.

Operation - 'metrics' or 'heatmap'
```bash
python test.py --image_dir_a datasets/{dataset}/results/targets
    --image_dir_b datasets/{dataset}/results/outputs
    --output_dir datasets/{dataset}/results --operation {operation}
```

## Acknowledgements
This implementation is based on a port of [pix2pix](https://github.com/phillipi/pix2pix) from PyTorch to Tensorflow which was written by [affinelayer](https://github.com/affinelayer/pix2pix-tensorflow). 
