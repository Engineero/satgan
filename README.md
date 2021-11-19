# SATGAN

*TODO: update the article URL once published.*

[Article about this implemention][0]

The self-attentive task generative adversarial network (SATGAN) learns
to emulate realistic target sensor noise characteristics in order to
augment existing datasets with simulated scenes that better approximate
real-world systems. It learns a mapping from random input noise to realistic
target-domain sensor characteristics while maintaining semantic information
in simulated scenes through the use of a task network. Example real images of
a space domain awareness (SDA) scene from the original paper are shown below:

![Real images](satgan_docs/real_images.png 'Real images')

Example noiseless simulated scenes used as context are below:

![Context images](satgan_docs/context_images.png 'Context images')

Finally example simulated scenes with generated addative noise are shown below:

![Fake images](satgan_docs/fake_images.png 'Fake images')

SATGAN comprises three parts: a generator based on a U-net implementation, a
discriminator based on PatchGAN, and a task network based on
[[Fletcher *et al.*][1]]. The SATGAN architecture is illustrated below:

![SATGAN architecture](satgan_docs/satgan_architecture.png 'SATGAN architecture')

## Setup

### Prerequisites
- Tensorflow >= 2.2.1
- Tensorflow-addons >= 0.11.2 (for optional mish activation)
- MISS YOLOv3

### Recommended
- Linux with Tensorflow GPU edition + cuDNN

### Getting Started

```sh
# clone this repo
git clone https://github.com/Engineero/satgan.git
cd satgan

# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)
python train_satgan.py \
  --mode train \
  --output_dir model_train \
  --max_epochs 200 \
  --input_dir my_data/train \
```

## Citation

*TODO: update paper link*

If you use this code for your research, please cite the paper this code is based on:
[Self-attending task generative adversarial network for realistic satellite image creation][0]:

```
@article{toner_self-attending_2021,
	title = {Self-{Attending} {Task} {Generative} {Adversarial} {Network} for {Realistic} {Satellite} {Image} {Creation}},
	url = {https://arxiv.org/abs/2111.09463v1},
	language = {en},
	urldate = {2021-11-19},
	author = {Toner, Nathan and Fletcher, Justin},
	month = nov,
	year = {2021},
	file = {Snapshot:/Users/nathantoner/Zotero/storage/K7AHTQEU/2111.html:text/html},
}
```

## Acknowledgments

[0]: https://www.google.com
[1]: https://amostech.com/TechnicalPapers/2019/Machine-Learning-for-SSA-Applications/Fletcher.pdf
