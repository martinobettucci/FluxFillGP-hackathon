01/16/25 : New version with its own model snapshots

# FLUX FILL
by Black Forest Labs: https://blackforestlabs.ai. Documentation for our API can be found here: [docs.bfl.ml](https://docs.bfl.ml/).
![grid](assets/grid.jpg)

## Flux Fill bug fixing, improvements and support for memory poor GPUs (less than 12GB of VRAM) by Deepbeepmeep.
This fork of Flux Fill (https://github.com/black-forest-labs/flux) is an illustration on how one can set up on an existing model some fast and properly working CPU offloading with very few changes in the core model.

For more information on how to use the mmpg module, please go to: https://github.com/deepbeepmeep/mmgp


Beside the support for 12 GB VRAM GPU with fast generation of images, I did a few improvements to the Flux Fill tool:
- bug fixing
- progression bar
- automatic resizing of large images
- user interface streamlined


Once the installation is done (see instructions below), run the Flux Fill tool with the command:

```bash
streamlit run demo_st_fill.py
```

## Configuration  

A minimum of 48 GB in your RAM is needed to run this tool.To reduce the RAM requirements to 32 GB and / or run the application even faster (for a small loss in quality), you may download directly a prequantized text encoder by changing the file *flux/conditioner.py* on line 21 to use a quantized model.

Alternatively if you have 64 GB of RAM, you can increase the quality of the image generation by using a non quantized Flux transformer model by changing the file *flux/util.py* on 333 to use a non quantized transformer model.

If you have more than 24 GB of VRAM you can set the option *profile = 1* on line 88 of file *demo_st_fill.py* for much faster generations.

## Local installation

```bash
cd $HOME && git clone https://github.com/black-forest-labs/flux
cd $HOME/flux
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124  # or a lower version of Cuda if it is not supported by your system
pip install -e ".[all]"
```

### Other Models for the GPU Poor
- HuanyuanVideoGP: https://github.com/deepbeepmeep/HunyuanVideoGP
One of the best open source Text to Video generator

- Cosmos1GP: https://github.com/deepbeepmeep/Cosmos1GP
This application include two models: a text to world generator and a image / video to world (probably the best open source image to video generator).



