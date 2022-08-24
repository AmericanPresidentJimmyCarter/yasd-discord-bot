# Yet Another Stable Diffusion Discord Bot

## Changelog

- 2022-08-24: Added k_lms and other k-diffusion samplers, with k-lms now the default. DDIM is still electable with "(sampler=ddim)" argument.

## Installation

**Coming soon: docker image!**

This installation is intended for debian or arch flavored linux users. YMMV. You will need to have Python 3 and pip installed.

Clone my SD fork of [dalle-flow](https://github.com/AmericanPresidentJimmyCarter/dalle-flow/tree/stable-diffusion) and checkout the right branch with the following commands:

```bash
git clone https://github.com/AmericanPresidentJimmyCarter/dalle-flow/
git checkout stable-diffusion
```

Then follow the instructions in the above [link](https://github.com/AmericanPresidentJimmyCarter/dalle-flow/tree/stable-diffusion) to install. DO NOT use the prebuilt docker image but instead follow the steps under "**Run natively**".

At this time, if you haven't already, you will need to put the stable diffusion weights into `stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt`.

Need to download the weights? Go to [Huggingface's repository for the latest version](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt), log in, agree to the terms and conditions, then download `sd-v1-4.ckpt`. Rename that to `model.ckpt` and put it into the location specified above.

To start jina with old models disabled when you're all done:

```bash
python flow_parser.py --enable-stable-diffusion --disable-dalle-mega --disable-glid3xl
jina flow --uses flow.tmp.yml
```

Jina should display lots of pretty pictures to tell you it's working. It may take a bit on first boot to load everything.

Somewhere else, clone this repository and follow these steps:

```bash
git clone https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/
cd yasd-discord-bot
mkdir image_docarrays
mkdir images
mkdir temp_json
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Then you can start the bot with:

```bash
python bot.py YOUR_DISCORD_BOT_TOKEN
```

**Be sure you have the "Message Content Intent" flag set to be on in your bot settings!**

Where YOUR_DISCORD_BOT_TOKEN is your [token](https://discordpy.readthedocs.io/en/stable/discord.html). Once the bot is connected, you can read about how to use it with `>help`.

The bot uses the folders as a bus to store/shuttle data. All images created are stored in `images/`.

OPTIONAL: If you aren't running jina on the same box, you will need change the address to connect to declared as constant `JINA_SERVER_URL` in `imagetool.py`.

## What can it do?

- Generate images from text (`>image foo bar`)
- Generate images from text with a frozen seed and variations in array format (`>image [foo, bar]`)
- Generate images from text while exploring seeds (`>image foo bar (seed_search=t)`)
- Generate images from images (and optionally prompts) (`>image2image foo bar`)
- Diffuse ("riff") on images it has previously generated (`riff <id> <idx>`)

Examples:

> ``` >image A United States twenty dollar bill with [Jerry Seinfeld, Jason Alexander, Michael Richards, Julia Louis-Dreyfus]'s portrait in the center (seed=2)```

![Seinfeld actors on money](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/blob/master/examples/seinfeld_money.jpg?raw=true)

> ```>image2image Still from Walt Disney's The Princess and the Frog, 2001 (iterations=4, strength=0.6, scale=15)```

Attached image

![Vermeer's Girl with a Pearl Earring](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/blob/master/examples/pearl.jpg?raw=true)

Output image

![Vermeer's Girl with a Pearl Earring diffused into Disney's Princess and the Front](https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/blob/master/examples/princess_frog.jpg?raw=true)


## What do I need?

An NVIDIA GPU with >= 16 GB of VRAM unless you turn down the Stable Diffusion resolution.

## Something is broken

Open an issue here.

## Closing remarks

Be cool, stay in school.

## License

[MIT](https://choosealicense.com/licenses/mit/)