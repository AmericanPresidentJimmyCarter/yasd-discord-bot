# Yet Another Stable Diffusion Discord Bot

![Yet Another Stable Diffusion Discord Bot Splash Image](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/blob/master/examples/yasd.jpg?raw=true)

Live now on the LAION Discord Server for you to try!

[![LAION Discord Server](https://discordapp.com/api/guilds/823813159592001537/widget.png?style=banner2)](https://discord.com/invite/UxX8dv5KMh)

## Features

- **Highly Scalable**: Leverages `dalle-flow` gRPC interface to independently serve images from any number of GPUs, while higher memory calls to the gRPC through the bot are forked onto individual instances of Python.
- **Support For Other Popular Models**: Latent diffusion GLID3XL or DALLE-MEGA can easily by turned on in addition to Stable Diffusion through `dalle-flow` for text-to-image generation.
- **Support For Low VRAM GPUS**: Stable Diffusion Lite supports image generation with GPUs >= 5 GB.
- **Supports Slash and Legacy Style Commands**: While Discord is moving towards the new slash style commands that feature auto-completion functions, YASD Discord Bot also features direct commands prefixed with `>` -- whichever you find easier.
- **Easy User Interface Including Buttons and Loading Indicators**: Riffing and upscaling your creations has never been easier! It even comes with a [manual](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/tree/master/manual#readme)!
- **Stores All Images and Prompts by Default**: Never lose your previous generations!

## Contents

- [Changelog](#changelog)
- [Content advisory](#content-advisory)
- [What do I need?](#what-do-i-need)
- [Installation](#installation)
  - [Docker Image](#docker-installation-docker-image)
  - [Docker Self-Build](#docker-installation-build-docker-image-yourself)
  - [Native](#native-installation)
- [What can it do?](#what-can-it-do)
- [User Manual](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/tree/master/manual#readme)
- [Something is broken](#something-is-broken)
- [Closing Remarks](#closing-remarks)
- [License](#license)


## Changelog

- 2022-09-11: Add optional NSFW spoiler filter and NSFW wordlist filter. Added the ability to set default steps and queue any quantity of images on a per user basis with a new flag.
- 2022-09-06: Added the ability to change make images of any size and riff into different sizes ("outriffing").
- 2022-09-05: The `sd-lite` branch has been merged upstream, so now low VRAM is available with docker images too.
- 2022-08-30: `optimized-sd` branch has moved to `sd-lite` branch, which will be merged upstream. Includes small bugfixes and enhanced interpolation. Upstream docker image is now functional, so instructions have been added for installing that.
- 2022-08-30: Updated to add slash commands in addition to legacy commands, added a manual link instead of help, added multi-user support (more than one user may now use the bot at a time without waiting), added `interpolate` command.
- 2022-08-28: Add ability to use with low VRAM cards through optimized `dalle-flow` branch `optimized-sd`.
- 2022-08-27: Add content advisory.
- 2022-08-26: Stable Diffusion branch merged into upstream `dalle-flow`. Added docker installation instructions.
- 2022-08-24: Added k_lms and other k-diffusion samplers, with k-lms now the default. DDIM is still electable with "(sampler=ddim)" argument.


## Content advisory

This bot does not come equipped with a NSFW filter for content by default and will make any content out of the box. Please be sure to read and agree with the [license for the weights](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE), as well as the [MIT license](https://en.wikipedia.org/wiki/MIT_License), and abide by all applicable laws and regulations in your respective area.

To enable the NSFW filter to automatically add the spoiler tag to any potential NSFW images, use the flag `--nsfw-auto-spoiler`. You must first `pip install -r requirements_nsfw_filter.txt` to get the modules required for this.

To enable NSFW prompt detection via BERT, use the flag `--nsfw-prompt-detection` and be sure to `pip install -r requirements_nsfw_filter.txt`.

To reject any prompts if they contain a word within a wordlist, use the `--nsfw-wordlist` flag, e.g. `--nsfw-wordlist bad_words.txt`. The wordlist should be strings separated by newlines.


## What do I need?

Python 3 3.9+ with pip and virtualenv installed (Ubuntu 22.04 works great!)

CUDA runtime environment installed

An NVIDIA GPU with >= 16 GB of VRAM (9GB if you disable SwinIR)

OR

An NVIDIA GPU with >= 5 GB of VRAM using the lite option

If running with a low VRAM GPU, you will not have access to the `>upscale` endpoint and will not have the ability to use multiple samplers. Pay close attention to all steps labeled **LOW VRAM GPU USERS** in the installation instructions.


## Installation

This installation is intended for debian or arch flavored linux users. YMMV. You will need to have Python 3 and pip installed.

```bash
sudo apt install python3 python3-pip
sudo pip3 install virtualenv
```


### Docker installation (docker image)

Install the [Nvidia docker container environment](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) if you have not already.


Pull the `dalle-flow` docker image with:

```bash
docker pull jinaai/dalle-flow:latest
```

Go to [Huggingface's repository for the latest version](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original), log in, agree to the terms and conditions, then download `sd-v1-4.ckpt`. Rename that to `model.ckpt` and then, from that directory, run the following commands:

```bash
mkdir ~/ldm
mkdir ~/ldm/stable-diffusion-v1
mv model.ckpt ~/ldm/stable-diffusion-v1/model.ckpt
```

Then run the container with this command:

```
sudo docker run -e DISABLE_CLIP="1" \
  -e DISABLE_DALLE_MEGA="1" \
  -e DISABLE_GLID3XL="1" \
  -e ENABLE_STABLE_DIFFUSION="1" \
  -p 51005:51005 \
  -it \
  -v ~/ldm:/dalle/stable-diffusion/models/ldm/ \
  -v $HOME/.cache:/home/dalle/.cache \
  --gpus all \
  jinaai/dalle-flow
```

***

**LOW VRAM GPU USERS**

Run this instead to enable the lite version and disable the SWINIR, which will crash your GPU. the `>upscale` endpoint will not be available.

```bash
sudo docker run -e DISABLE_CLIP="1" \
  -e DISABLE_DALLE_MEGA="1" \
  -e DISABLE_GLID3XL="1" \
  -e DISABLE_SWINIR="1" \
  -e ENABLE_STABLE_DIFFUSION_LITE="1" \
  -p 51005:51005 \
  -it \
  -v ~/ldm:/dalle/stable-diffusion/models/ldm/ \
  -v $HOME/.cache:/home/dalle/.cache \
  --gpus all \
  jinaai/dalle-flow
```

If you have >= 12 GB of VRAM, you can re-enable SWINIR.

***

Somewhere else, clone this repository and follow these steps:

```bash
git clone https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/
cd yasd-discord-bot
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Then you can start the bot with:

```bash
python bot.py YOUR_DISCORD_BOT_TOKEN -g YOUR_GUILD_ID
```

**LOW VRAM GPU USERS**: You can disable the other samplers showing up, which do nothing, by adding the flag `--optimized-sd` to the above command.

**Be sure you have the "Message Content Intent" flag set to be on in your bot settings!**

Where YOUR_DISCORD_BOT_TOKEN is your [token](https://discordpy.readthedocs.io/en/stable/discord.html) and YOUR_GUILD_ID is the integer ID for your server (right click on the server name, then click "Copy ID"). Supplying the guild ID is optional, but it will result in the slash commands being available to your server almost instantly. Once the bot is connected, you can read about how to use it with `>help`.

The bot uses the folders as a bus to store/shuttle data. All images created are stored in `images/`.

OPTIONAL: If you aren't running jina on the same box, you will need change the address to connect to declared as constant `JINA_SERVER_URL` in `imagetool.py`.


### Docker installation (build docker image yourself)

Install the [Nvidia docker container environment](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) if you have not already.

Make a folder for `dalle-flow`:

```bash
mkdir ~/dalle
cd ~/dalle
git clone https://github.com/jina-ai/dalle-flow
cd dalle-flow
```

Go to [Huggingface's repository for the latest version](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original), log in, agree to the terms and conditions, then download `sd-v1-4.ckpt`. Rename that to `model.ckpt` and then, from that directory, run the following commands:

```bash
mkdir ~/ldm
mkdir ~/ldm/stable-diffusion-v1
mv model.ckpt ~/ldm/stable-diffusion-v1/model.ckpt
```

In the `dalle-flow` folder (`cd ~/dalle/dalle-flow`), build with this command:

```bash
docker build --build-arg GROUP_ID=$(id -g ${USER}) --build-arg USER_ID=$(id -u ${USER}) -t jinaai/dalle-flow .
```

Then run the container with this command:

```
sudo docker run -e DISABLE_CLIP="1" \
  -e DISABLE_DALLE_MEGA="1" \
  -e DISABLE_GLID3XL="1" \
  -e ENABLE_STABLE_DIFFUSION="1" \
  -p 51005:51005 \
  -it \
  -v ~/ldm:/dalle/stable-diffusion/models/ldm/ \
  -v $HOME/.cache:/home/dalle/.cache \
  --gpus all \
  jinaai/dalle-flow
```

***

**LOW VRAM GPU USERS**

Run this instead to enable the lite version and disable the SWINIR, which will crash your GPU. the `>upscale` endpoint will not be available.

```bash
sudo docker run -e DISABLE_CLIP="1" \
  -e DISABLE_DALLE_MEGA="1" \
  -e DISABLE_GLID3XL="1" \
  -e DISABLE_SWINIR="1" \
  -e ENABLE_STABLE_DIFFUSION_LITE="1" \
  -p 51005:51005 \
  -it \
  -v ~/ldm:/dalle/stable-diffusion/models/ldm/ \
  -v $HOME/.cache:/home/dalle/.cache \
  --gpus all \
  jinaai/dalle-flow
```

If you have >= 12 GB of VRAM, you can re-enable SWINIR.

***

Somewhere else, clone this repository and follow these steps:

```bash
git clone https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/
cd yasd-discord-bot
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Then you can start the bot with:

```bash
python bot.py YOUR_DISCORD_BOT_TOKEN -g YOUR_GUILD_ID
```

**LOW VRAM GPU USERS**: You can disable the other samplers showing up, which do nothing, by adding the flag `--optimized-sd` to the above command.

**Be sure you have the "Message Content Intent" flag set to be on in your bot settings!**

Where YOUR_DISCORD_BOT_TOKEN is your [token](https://discordpy.readthedocs.io/en/stable/discord.html) and YOUR_GUILD_ID is the integer ID for your server (right click on the server name, then click "Copy ID"). Supplying the guild ID is optional, but it will result in the slash commands being available to your server almost instantly. Once the bot is connected, you can read about how to use it with `>help`.

The bot uses the folders as a bus to store/shuttle data. All images created are stored in `images/`.

OPTIONAL: If you aren't running jina on the same box, you will need change the address to connect to declared as constant `JINA_SERVER_URL` in `imagetool.py`.

### Native installation

Follow the instructions for [dalle-flow](https://github.com/jina-ai/dalle-flow) to install and run that server. The steps you need to follow can be found under "**Run natively**". Once `flow` is up and running, proceed to the next step.

At this time, if you haven't already, you will need to put the stable diffusion weights into `dalle/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt`.

Need to download the weights? Go to [Huggingface's repository for the latest version](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original), log in, agree to the terms and conditions, then download `sd-v1-4.ckpt`. Rename that to `model.ckpt` and put it into the location specified above.

To start jina with old models disabled when you're all done:

```bash
python flow_parser.py --enable-stable-diffusion --disable-dalle-mega --disable-glid3xl
jina flow --uses flow.tmp.yml
```

***

**LOW VRAM GPU USERS**

Run this instead to disable the SWINIR, which will crash your GPU. the `>upscale` endpoint will not be available.

```bash
python flow_parser.py --disable-clip --enable-stable-diffusion-lite --disable-dalle-mega --disable-glid3xl --disable-swinir
jina flow --uses flow.tmp.yml
```

If you have >= 12 GB of VRAM, you can re-enable SWINIR.

***

Jina should display lots of pretty pictures to tell you it's working. It may take a bit on first boot to load everything.

Somewhere else, clone this repository and follow these steps:

```bash
git clone https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/
cd yasd-discord-bot
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Then you can start the bot with:

```bash
python bot.py YOUR_DISCORD_BOT_TOKEN -g YOUR_GUILD_ID
```

**LOW VRAM GPU USERS**: You can disable the other samplers showing up, which do nothing, by adding the flag `--optimized-sd` to the above command.

**Be sure you have the "Message Content Intent" flag set to be on in your bot settings!**

Where YOUR_DISCORD_BOT_TOKEN is your [token](https://discordpy.readthedocs.io/en/stable/discord.html) and YOUR_GUILD_ID is the integer ID for your server (right click on the server name, then click "Copy ID"). Supplying the guild ID is optional, but it will result in the slash commands being available to your server almost instantly. Once the bot is connected, you can read about how to use it with `>help`.

The bot uses the folders as a bus to store/shuttle data. All images created are stored in `images/`.

OPTIONAL: If you aren't running jina on the same box, you will need change the address to connect to declared as constant `JINA_SERVER_URL` in `imagetool.py`.

## What can it do?

- Generate images from text (`/image foo bar`)
- Generate images from text with a frozen seed and variations in array format (`/image [foo, bar]`)
- Generate images from text while exploring seeds (`/image foo bar (seed_search=t)`)
- Generate images from images (and optionally prompts) (`>image2image foo bar`)
- Diffuse ("riff") on images it has previously generated (`/riff <id> <idx>`)
- Interpolate between two prompts (`/interpolate <prompt 1> <prompt 2>`)

Examples:

> ``` >image A United States twenty dollar bill with [Jerry Seinfeld, Jason Alexander, Michael Richards, Julia Louis-Dreyfus]'s portrait in the center (seed=2)```

![Seinfeld actors on money](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/blob/master/examples/seinfeld_money.jpg?raw=true)

> ```>image2image Still from Walt Disney's The Princess and the Frog, 2001 (iterations=4, strength=0.6, scale=15)```

Attached image

![Vermeer's Girl with a Pearl Earring](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/blob/master/examples/pearl.jpg?raw=true)

Output image

![Vermeer's Girl with a Pearl Earring diffused into Disney's Princess and the Front](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/blob/master/examples/princess_frog.jpg?raw=true)


> ```>interpolate Sonic the Hedgehog portrait | European Hedgehog Portrait```

![Interpolation of Sonic the Hedgehog portrait to European Hedgehog Portrait](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/blob/master/examples/interpolate.jpg?raw=true)

## Something is broken

Open an issue here.

## Closing remarks

Be cool, stay in school.

## License

Copyright 2022 Jimmy (AmericanPresidentJimmyCarter)

[MIT](https://choosealicense.com/licenses/mit/)
