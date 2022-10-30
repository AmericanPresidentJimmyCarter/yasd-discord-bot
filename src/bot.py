import argparse
import json
import pathlib
import sys
import time

from io import BytesIO
from typing import Callable, List, Optional, Union
from urllib.error import URLError
from urllib.request import urlopen

import discord

from PIL import Image
from discord import app_commands
from docarray import Document, DocumentArray
from tqdm import tqdm

import actions
from client import YASDClient
from constants import (
    DOCARRAY_LOCATION_FN,
    DOCARRAY_STORAGE_FOLDER,
    HEIGHT_AND_WIDTH_CHOICES,
    IMAGE_LOCATION_FN,
    IMAGE_STORAGE_FOLDER,
    MANUAL_LINK,
    MAX_ITERATIONS,
    MAX_SCALE,
    MAX_SEED,
    MAX_STEPS,
    MAX_STRENGTH,
    MIN_ITERATIONS,
    MIN_SCALE,
    MIN_STEPS,
    MIN_STRENGTH,
    NUM_IMAGES_MAX,
    OUTPAINT_CHOICES,
    SAMPLER_CHOICES,
    TEMP_JSON_STORAGE_FOLDER,
    UPSCALER_CHOICES,
    UPSCALER_NONE,
    UPSCALER_REALESRGAN_4X,
    UPSCALER_REALESRGAN_4X_ANIME,
    UPSCALER_REALESRGAN_4X_FACE,
    UPSCALER_SWINIR,
    VALID_IMAGE_HEIGHT_WIDTH,
    VALID_SAMPLERS,
)
from ui import (
    FourImageButtons,
)
from util import (
    prompt_contains_nsfw,
    prompt_has_valid_sd_custom_embeddings,
    resize_image,
    short_id_generator,
)


parser = argparse.ArgumentParser()
parser.add_argument('token', help='Discord token')
parser.add_argument('--allow-queue', dest='allow_queue',
    action=argparse.BooleanOptionalAction)
parser.add_argument('--default-sampler', dest='default_sampler', nargs='?',
    type=str, help='Default sampler to use', default=None)
parser.add_argument('--default-steps', dest='default_steps', nargs='?',
    type=int, help='Default number of steps for the sampler', default=50)
parser.add_argument('--hours-on-server-to-use', dest='hours_needed', nargs='?',
    type=int,
    help='The hours the user has been on the server before they can use the bot',
    required=False)
parser.add_argument('-g', '--guild', dest='guild',
    help='Discord guild ID', type=int, required=False)
parser.add_argument('--max-queue',
    dest='max_queue',
    type=int,
    help='The maximum number of simultaneous requests per user',
    required=False,
    default=9999,
)
parser.add_argument('--nsfw-auto-spoiler', dest='auto_spoiler',
    action=argparse.BooleanOptionalAction)
parser.add_argument('--nsfw-prompt-detection',
    dest='nsfw_prompt_detection',
    action=argparse.BooleanOptionalAction)
parser.add_argument('--nsfw-wordlist',
    dest='nsfw_wordlist',
    help='Newline separated wordlist filename',
    type=str,
    required=False)
parser.add_argument('--reload-last-minutes', dest='reload_last_minutes',
    help='When reloading the bot, how far back in minutes to load old ' +
    'UI elements (default 120 minutes)', type=int, required=False)
parser.add_argument('--restrict-all-to-channel',
    dest='restrict_all_to_channel',
    help='Restrict all commands to a specific channel',
    type=int, required=False)
parser.add_argument('--restrict-slash-to-channel',
    dest='restrict_slash_to_channel',
    help='Restrict slash commands to a specific channel',
    type=int, required=False)
args = parser.parse_args()


# Load up diffusers NSFW detection model and the NSFW wordlist detector.
nsfw_toxic_detection_fn: Callable|None = None
nsfw_wordlist: list[str] = []
safety_feature_extractor: Callable|None = None
safety_checker: Callable|None = None
if args.auto_spoiler:
    from diffusers.pipelines.stable_diffusion.safety_checker import (
        StableDiffusionSafetyChecker,
    )
    from transformers import AutoFeatureExtractor

    # load safety model
    SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
        SAFETY_MODEL_ID)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        SAFETY_MODEL_ID)
if args.nsfw_prompt_detection:
    from detoxify import Detoxify
    nsfw_toxic_detection_fn = Detoxify('multilingual').predict
if args.nsfw_wordlist:
    with open(args.nsfw_wordlist, 'r') as lst_f:
        nsfw_wordlist = lst_f.readlines()
        nsfw_wordlist = [word.strip().lower() for word in nsfw_wordlist]


async def prompt_check_fn(
    prompt: str,
    author_id: str,
    channel: discord.abc.GuildChannel,
) -> str|bool:
    '''
    Check if a prompt is valid and return either the prompt or False if it is
    not valid.
    '''
    if (args.nsfw_wordlist or nsfw_toxic_detection_fn is not None) and \
        prompt_contains_nsfw(prompt, nsfw_toxic_detection_fn, nsfw_wordlist):
        await channel.send('Sorry, this prompt potentially contains NSFW ' +
            'or offensive content.')
        return False

    try:
        prompt_has_valid_sd_custom_embeddings(prompt)
    except Exception:
        await channel.send('Sorry, one of your custom embeddings is invalid.')
        return False

    return prompt

if args.default_sampler is not None and \
    args.default_sampler not in VALID_SAMPLERS:
    print(f'Invalid sampler "{args.default_sampler}" was given. Please use one ' +
        f'of {VALID_SAMPLERS}.')
    sys.exit(1)

guild = args.guild

# In memory k-v stores.
currently_fetching_ai_image: dict[str, Union[str, List[str], bool]] = {}
user_image_generation_nonces: dict[str, int] = {}

BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY = 'four_img_views'
button_store_dict: dict[str, list] = { BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY: [] }


BUTTON_STORE = f'{TEMP_JSON_STORAGE_FOLDER}/button-store-{str(guild)}.json'


pathlib.Path(DOCARRAY_STORAGE_FOLDER).mkdir(parents=True, exist_ok=True)
pathlib.Path(IMAGE_STORAGE_FOLDER).mkdir(parents=True, exist_ok=True)
pathlib.Path(TEMP_JSON_STORAGE_FOLDER).mkdir(parents=True, exist_ok=True)


# A simple JSON store for button views that we write to when making new buttons
# for new calls and which is kept in memory and keeps track of all buttons ever
# added. This allows us to persist buttons between reboots of the bot, power
# outages, etc.
#
# TODO Clear out old buttons on boot, since we ignore anything more than 48
# hours old below anyway.
bs_path = pathlib.Path(BUTTON_STORE)
if bs_path.is_file():
    with open(bs_path, 'r') as bs:
        button_store_dict = json.load(bs)


intents = discord.Intents(
    messages=True,
    dm_messages=True,
    guild_messages=True,
    message_content=True,
)


client = YASDClient(
    button_store_dict=button_store_dict,
    button_store_path=bs_path,
    cli_args=args,
    currently_fetching_ai_image=currently_fetching_ai_image,
    guild_id=guild,
    intents=intents,
    prompt_check_fn=prompt_check_fn,
    safety_checker=safety_checker,
    safety_feature_extractor=safety_feature_extractor,
    user_image_generation_nonces=user_image_generation_nonces,
)


@client.tree.command(
    description='Create an image from a prompt. ' +
        'Variations are given in square brackets ' +
        'e.g. "a [red, blue] ball"',
)
@app_commands.describe(
    height='Height of the image (default=512)',
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default=k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0, default=7.5)',
    seed='Deterministic seed for prompt (1 to 2^32-1, default=random)',
    seed_search='Seed searching mode, enumerates 9 different seeds starting at given seed (default=False)',
    steps=f'Number of steps to perform (10 to 250, default={args.default_steps})',
    width='Width of the image (default=512)',
)
@app_commands.choices(
    height=HEIGHT_AND_WIDTH_CHOICES,
    sampler=SAMPLER_CHOICES,
    width=HEIGHT_AND_WIDTH_CHOICES,
)
async def image(
    interaction: discord.Interaction,

    prompt: str,
    height: Optional[app_commands.Choice[int]] = None,
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, 1.0, 50.0]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    seed_search: Optional[bool]=False,
    steps: Optional[app_commands.Range[int, MIN_STEPS, MAX_STEPS]] = None,
    width: Optional[app_commands.Choice[int]] = None,
):
    await interaction.response.defer(thinking=True)

    if args.restrict_slash_to_channel:
        if interaction.channel.id != args.restrict_slash_to_channel:
            await interaction.followup.send('You are not allowed to use this in this channel!')
            return

    sid = await actions.image(
        interaction.channel, interaction.user, client, prompt,
        height=height.value if height is not None else None,
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
        seed_search=bool(seed_search),
        steps=steps,
        width=width.value if width is not None else None)
    if sid is not None:
        await interaction.followup.send(sid)
    else:
        await interaction.followup.send('Failed!')


@client.tree.command(
    description='Create an image from a generated image using its ID and an index',
)
@app_commands.describe(
    docarray_id='The ID for the bot-generated image you want to riff',
    height='Height of the image (default=512)',
    idx='The index of the bot generated image you want to riff',
    iterations='Number of diffusion iterations (1 to 16, default=1)',
    latentless='Do not compute latent embeddings from original image (default=False)',
    outpaint='Extend the image in various directions',
    prompt='Prompt, which overrides the original prompt for the image (default=None)',
    prompt_mask='Prompt to generate a mask to inpaint one, add "not " prefix to invert (default=None)',
    resize='Resize the image when adjusting width/height instead of attempting outpaint (default=False)',
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default=k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0, default=7.5)',
    seed='Deterministic seed for prompt (1 to 2^32-1, default=random)',
    strength="Strength of conditioning (0.01 <= strength <= 0.99, default=0.75)",
    width='Width of the image (default=512)',
)
@app_commands.choices(
    height=HEIGHT_AND_WIDTH_CHOICES,
    outpaint=OUTPAINT_CHOICES,
    sampler=SAMPLER_CHOICES,
    width=HEIGHT_AND_WIDTH_CHOICES,
)
async def riff(
    interaction: discord.Interaction,

    docarray_id: str,
    idx: app_commands.Range[int, 0, NUM_IMAGES_MAX-1],

    height: Optional[app_commands.Choice[int]] = None,
    iterations: Optional[app_commands.Range[int, MIN_ITERATIONS, MAX_ITERATIONS]] = None,
    latentless: Optional[bool]=False,
    outpaint: Optional[app_commands.Choice[str]] = None,
    prompt: Optional[str]=None,
    prompt_mask: Optional[str]=None,
    resize: Optional[bool]=False,
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, MIN_SCALE, MAX_SCALE]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    steps: Optional[app_commands.Range[int, MIN_STEPS, MAX_STEPS]] = None,
    strength: Optional[app_commands.Range[float, MIN_STRENGTH, MAX_STRENGTH]] = None,
    width: Optional[app_commands.Choice[int]] = None,
):
    await interaction.response.defer(thinking=True)

    if args.restrict_slash_to_channel:
        if interaction.channel.id != args.restrict_slash_to_channel:
            await interaction.followup.send('You are not allowed to use this in this channel!')
            return

    sid = await actions.riff(
        interaction.channel, interaction.user, client, docarray_id, idx,
        height=height.value if height is not None else None,
        iterations=iterations,
        latentless=bool(latentless),
        outpaint_mode=outpaint.value if outpaint is not None else None,
        prompt=prompt,
        prompt_mask=prompt_mask,
        resize=bool(resize),
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
        steps=steps,
        strength=strength,
        width=width.value if width is not None else None)
    if sid is not None:
        await interaction.followup.send(sid)
    else:
        await interaction.followup.send('Failed!')


@client.tree.command(
    description='Create an image from a URL',
)
@app_commands.describe(
    url='URL for an image',
    prompt='Prompt to use for riffing the image',
    height='Height of the image (default=512)',
    iterations='Number of diffusion iterations (1 to 16, default=1)',
    latentless='Do not compute latent embeddings from original image (default=False)',
    prompt_mask='Prompt to generate a mask to inpaint one (default=None)',
    resize='Resize the image when adjusting width/height instead of attempting outpaint (default=False)',
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default=k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0, default=7.5)',
    seed='Deterministic seed for prompt (1 to 2^32-1, default=random)',
    strength="Strength of conditioning (0.01 <= strength <= 0.99, default=0.75)",
    width='Width of the image (default=512)',
)
@app_commands.choices(
    height=HEIGHT_AND_WIDTH_CHOICES,
    sampler=SAMPLER_CHOICES,
    width=HEIGHT_AND_WIDTH_CHOICES,
)
async def image2image(
    interaction: discord.Interaction,

    url: str,
    prompt: str,

    height: Optional[app_commands.Choice[int]] = None,
    iterations: Optional[app_commands.Range[int, MIN_ITERATIONS, MAX_ITERATIONS]] = None,
    latentless: Optional[bool]=False,
    prompt_mask: Optional[str]=None,
    resize: Optional[bool]=False,
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, MIN_SCALE, MAX_SCALE]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    steps: Optional[app_commands.Range[int, MIN_STEPS, MAX_STEPS]] = None,
    strength: Optional[app_commands.Range[float, MIN_STRENGTH, MAX_STRENGTH]] = None,
    width: Optional[app_commands.Choice[int]] = None,
):
    await interaction.response.defer(thinking=True)

    if args.restrict_slash_to_channel:
        if interaction.channel.id != args.restrict_slash_to_channel:
            await interaction.followup.send('You are not allowed to use this in this channel!')
            return

    image = None
    short_id = short_id_generator()
    try:
        url_data = urlopen(url)
        image = Image.open(BytesIO(url_data.read()))
    except URLError as e:
        await interaction.followup.send(f'Bad image URL "{url}": {str(e)}!')
        return
    except OSError as e:
        await interaction.followup.send('Failed to parse image!')
        return

    image_fn = IMAGE_LOCATION_FN(short_id)
    da_fn = DOCARRAY_LOCATION_FN(short_id)
    image = resize_image(image)
    image.save(image_fn, format='PNG')

    buffered = BytesIO()
    image.save(buffered, format='PNG')
    _d = Document(
        blob=buffered.getvalue(),
        mime_type='image/png',
        tags={
            'text': '',
            'generator': 'discord image upload',
            'request_time': int(time.time()),
        },
    ).convert_blob_to_datauri()
    _d.text = prompt
    da = DocumentArray([_d])
    da.save_binary(da_fn, protocol='protobuf', compress='lz4')

    await interaction.channel.send(f'URL "{url}" sent by ' +
        f'<@{str(interaction.user.id)}> has been uploaded and ' +
        f'given ID `{short_id}`.')

    sid = await actions.riff(
        interaction.channel, interaction.user, client, short_id, 0,
        height=height.value if height is not None else None,
        iterations=iterations,
        latentless=bool(latentless),
        prompt=prompt,
        prompt_mask=prompt_mask,
        resize=bool(resize),
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
        steps=steps,
        strength=strength,
        width=width.value if width is not None else None)
    if sid is not None:
        await interaction.followup.send(sid)
    else:
        await interaction.followup.send('Failed!')


@client.tree.command(
    description='Create an interpolation from two prompts',
)
@app_commands.describe(
    prompt1='The starting prompt',
    prompt2='The ending prompt',
    height='Height of the image (default=512)',
    resample_prior='Resample the prior images during interpolation rather than using a new one each time (default=True)',
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default=k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0, default=7.5)',
    seed='Deterministic seed for prompt (1 to 2^32-1, default=random)',
    strength="Strength of conditioning (0.01 <= strength <= 0.99, default=0.75)",
    width='Height of the image (default=512)',
)
@app_commands.choices(
    height=HEIGHT_AND_WIDTH_CHOICES,
    sampler=SAMPLER_CHOICES,
    width=HEIGHT_AND_WIDTH_CHOICES,
)
async def interpolate(
    interaction: discord.Interaction,
    prompt1: str,
    prompt2: str,

    height: Optional[app_commands.Choice[int]] = None,
    resample_prior: Optional[bool]=True,
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, MIN_SCALE, MAX_SCALE]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    steps: Optional[app_commands.Range[int, MIN_STEPS, MAX_STEPS]] = None,
    strength: Optional[app_commands.Range[float, MIN_STRENGTH, MAX_STRENGTH]] = None,
    width: Optional[app_commands.Choice[int]] = None,
):
    await interaction.response.defer(thinking=True)

    if args.restrict_slash_to_channel:
        if interaction.channel.id != args.restrict_slash_to_channel:
            await interaction.followup.send('You are not allowed to use this in this channel!')
            return

    sid = await actions.interpolate(
        interaction.channel, interaction.user, client, prompt1, prompt2,
        height=height.value if height is not None else None,
        resample_prior=bool(resample_prior),
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
        steps=steps,
        strength=strength,
        width=width.value if width is not None else None)
    if sid is not None:
        await interaction.followup.send(sid)
    else:
        await interaction.followup.send('Failed!')


@client.tree.command(
    description='Upscale a generated image using its ID and index',
)
@app_commands.describe(
    docarray_id='The ID for the bot-generated image you want to upscale',
    idx='The index of the bot generated image you want to upscale',
)
@app_commands.choices(
    upscaler=UPSCALER_CHOICES,
)
async def upscale(
    interaction: discord.Interaction,
    docarray_id: str,
    idx: app_commands.Range[int, 0, NUM_IMAGES_MAX-1],
    upscaler: Optional[app_commands.Choice[str]] = None,
):
    await interaction.response.defer(thinking=True)

    if args.restrict_slash_to_channel:
        if interaction.channel.id != args.restrict_slash_to_channel:
            await interaction.followup.send('You are not allowed to use this in this channel!')
            return

    await actions.upscale(
        interaction.channel, interaction.user, client, docarray_id, idx,
        upscaler=upscaler.value if upscaler is not None else None)
    await interaction.followup.send('Done!')


@client.event
async def on_message(message):
    '''
    The on_message handler exists for the sole purpose of letting users upload
    images to store and use to generate riffs as the old image2image pipeline
    along with being able to use all of the legacy commands.
    Discord sucks and decided it would be cool to make every bot move to the
    slash command without any way to upload things using a slash command.

    TODO: Support uploads through the slash command when it's available.
    '''
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>help'):
        await message.channel.send('Please refer to the usage manual here: ' +
            MANUAL_LINK)
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>image '):
        prompt = message.clean_content[7:]
        height = None
        sampler = None
        scale = None
        seed = None
        seed_search = False
        steps = args.default_steps
        width = None

        parens_idx = prompt.find('(')
        if parens_idx >= 0:
            text = prompt[parens_idx:]
            if len(text) > 0 and text[0] == '(' and text[-1] == ')':
                opts = {}
                try:
                    opts = { val.split('=')[0].strip(): val.split('=')[1].strip()
                        for val in text[1:-1].split(',') }
                except IndexError:
                    pass
                if 'sampler' in opts:
                    sampler = opts['sampler']
                if 'height' in opts:
                    try:
                        height_int = int(opts['height'])
                        if height_int in VALID_IMAGE_HEIGHT_WIDTH:
                            height = height_int
                    except Exception:
                        pass
                if 'scale' in opts:
                    try:
                        scale_float = float(opts['scale'])
                        if scale_float >= 0. and scale_float <= 50.:
                            scale = scale_float
                    except Exception:
                        pass
                if 'seed' in opts:
                    try:
                        seed_int = int(opts['seed'])
                        if seed_int >= 0 and seed_int <= MAX_SEED:
                            seed = seed_int
                    except Exception:
                        pass
                if 'seed_search' in opts:
                    seed_search = True
                if 'steps' in opts:
                    try:
                        steps_int = int(opts['steps'])
                        if steps_int >= MIN_STEPS and steps_int <= MAX_STEPS:
                            steps = steps_int
                    except Exception:
                        pass
                if 'width' in opts:
                    try:
                        width_int = int(opts['width'])
                        if width_int in VALID_IMAGE_HEIGHT_WIDTH:
                            width = width_int
                    except Exception:
                        pass
            prompt = prompt[0:parens_idx]
        await actions.image(message.channel, message.author, client, prompt,
            height, sampler, scale, seed, seed_search, steps, width)
        return
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>riff '):
        msg_split = message.clean_content.split(' ')
        if len(msg_split) < 3:
            await message.channel.send('Riff requires at least two arguments')
            return

        docarray_id = msg_split[1]
        idx = 0
        try:
            idx = int(msg_split[2])
        except Exception as e:
            pass

        text = ''
        if len(msg_split) > 3:
            text = ' '.join(msg_split[3:])

        height = None
        iterations = None
        latentless = False
        outpaint_mode = None
        prompt = None
        prompt_mask = None
        resize = False
        sampler = None
        scale = None
        seed = None
        steps = args.default_steps
        strength = None
        width = None
        if len(text) > 0 and text[0] == '(' and text[-1] == ')':
            opts = {}
            try:
                opts = { val.split('=')[0].strip(): val.split('=')[1].strip()
                    for val in text[1:-1].split(',') }
            except IndexError:
                pass

            if 'height' in opts:
                try:
                    height_int = int(opts['height'])
                    if height_int in VALID_IMAGE_HEIGHT_WIDTH:
                        height = height_int
                except Exception:
                    pass
            if 'outpaint_mode' in opts:
                outpaint_mode = opts['outpaint_mode']
            if 'prompt' in opts:
                prompt = opts['prompt']
            if 'prompt_mask' in opts:
                prompt_mask = opts['prompt_mask']
            if 'iterations' in opts:
                try:
                    iterations_int = int(opts['iterations'])
                    if iterations_int > 0 and iterations_int <= MAX_ITERATIONS:
                        iterations = iterations_int
                except Exception:
                    pass
            if 'latentless' in opts:
                latentless = True
            if 'resize' in opts:
                resize = True
            if 'sampler' in opts:
                sampler = opts['sampler']
            if 'scale' in opts:
                try:
                    scale_float = float(opts['scale'])
                    if scale_float >= 0. and scale_float <= 50.:
                        scale = scale_float
                except Exception:
                    pass
            if 'seed' in opts:
                try:
                    seed_int = int(opts['seed'])
                    if seed_int >= 0 and seed_int <= MAX_SEED:
                        seed = seed_int
                except Exception:
                    pass
            if 'steps' in opts:
                try:
                    steps_int = int(opts['steps'])
                    if steps_int >= MIN_STEPS and steps_int <= MAX_STEPS:
                        steps = steps_int
                except Exception:
                    pass
            if 'strength' in opts:
                try:
                    strength_float = float(opts['strength'])
                    if strength_float > 0. and strength_float < 1.:
                        strength = strength_float
                except Exception:
                    pass
            if 'width' in opts:
                try:
                    width_int = int(opts['width'])
                    if width_int in VALID_IMAGE_HEIGHT_WIDTH:
                        width = width_int
                except Exception:
                    pass

        await actions.riff(
            message.channel, message.author, client, docarray_id, idx,
            height=height,
            iterations=iterations,
            latentless=bool(latentless),
            outpaint_mode=outpaint_mode,
            prompt=prompt,
            prompt_mask=prompt_mask,
            resize=resize,
            sampler=sampler,
            scale=scale,
            seed=seed,
            steps=steps,
            strength=strength,
            width=width)
        return
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>image2image'):
        prompt = message.clean_content[13:]
        sid = short_id_generator()
        image_fn = IMAGE_LOCATION_FN(sid)
        da_fn = DOCARRAY_LOCATION_FN(sid)
        if len(message.attachments) != 1:
            await message.channel.send(
                'Please upload a single image with your message')
            return
        else:
            image_bytes = await message.attachments[0].read()
            try:
                image = Image.open(BytesIO(image_bytes))
            except Exception:
                import traceback
                traceback.print_exc()
                await message.channel.send(
                    f'Could not load image file for attachment {message.attachments[0].filename}')
                return

            image = resize_image(image)
            image.save(image_fn, format='PNG')

            buffered = BytesIO()
            image.save(buffered, format='PNG')
            _d = Document(
                blob=buffered.getvalue(),
                mime_type='image/png',
                tags={
                    'text': '',
                    'generator': 'discord image upload',
                    'request_time': int(time.time()),
                },
            ).convert_blob_to_datauri()
            _d.text = message.clean_content

            da = DocumentArray([_d])
            da.save_binary(da_fn, protocol='protobuf', compress='lz4')

        iterations = None
        latentless = False
        outpaint_mode = None
        prompt_mask = None
        resize = False
        sampler = None
        scale = None
        seed = None
        steps = args.default_steps
        strength = None

        parens_idx = prompt.find('(')
        if parens_idx >= 0:
            text = prompt[parens_idx:]
            if len(text) > 0 and text[0] == '(' and text[-1] == ')':
                opts = {}
                try:
                    opts = { val.split('=')[0].strip(): val.split('=')[1].strip()
                        for val in text[1:-1].split(',') }
                except IndexError:
                    pass

                if 'iterations' in opts:
                    try:
                        iterations_int = int(opts['iterations'])
                        if iterations_int > 0 and iterations_int <= MAX_ITERATIONS:
                            iterations = iterations_int
                    except Exception:
                        pass
                if 'latentless' in opts:
                    latentless = True
                if 'outpaint_mode' in opts:
                    outpaint_mode = opts['outpaint_mode']
                if 'prompt_mask' in opts:
                    prompt_mask = opts['prompt_mask']
                if 'resize' in opts:
                    resize = True
                if 'sampler' in opts:
                    sampler = opts['sampler']
                if 'scale' in opts:
                    try:
                        scale_float = float(opts['scale'])
                        if scale_float >= 0. and scale_float <= 50.:
                            scale = scale_float
                    except Exception:
                        pass
                if 'seed' in opts:
                    try:
                        seed_int = int(opts['seed'])
                        if seed_int >= 0 and seed_int <= MAX_SEED:
                            seed = seed_int
                    except Exception:
                        pass
                if 'steps' in opts:
                    try:
                        steps_int = int(opts['steps'])
                        if steps_int >= MIN_STEPS and steps_int <= MAX_STEPS:
                            steps = steps_int
                    except Exception:
                        pass
                if 'strength' in opts:
                    try:
                        strength_float = float(opts['strength'])
                        if strength_float > 0. and strength_float < 1.:
                            strength = strength_float
                    except Exception:
                        pass
            prompt = prompt[0:parens_idx]

        await actions.riff(message.channel, message.author, client, sid, 0,
            iterations=iterations,
            latentless=latentless,
            outpaint_mode=outpaint_mode,
            prompt=prompt,
            prompt_mask=prompt_mask,
            resize=resize,
            sampler=sampler,
            scale=scale,
            seed=seed,
            steps=steps,
            strength=strength)
        return
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>interpolate '):
        prompt = message.clean_content[13:]

        if '|' not in prompt:
            await message.channel.send('Pipe to separate prompts must be used ' +
                'with >interpolate')
            return

        height = None
        resample_prior = True
        sampler = None
        scale = None
        seed = None
        steps = args.default_steps
        strength = None
        width = None
        parens_idx = prompt.find('(')
        if parens_idx >= 0:
            text = prompt[parens_idx:]
            if len(text) > 0 and text[0] == '(' and text[-1] == ')':
                opts = {}
                try:
                    opts = { val.split('=')[0].strip(): val.split('=')[1].strip()
                        for val in text[1:-1].split(',') }
                except IndexError:
                    pass

                if 'height' in opts:
                    try:
                        height_int = int(opts['height'])
                        if height_int in VALID_IMAGE_HEIGHT_WIDTH:
                            height = height_int
                    except Exception:
                        pass
                if 'resample_prior' in opts and \
                    (
                        opts['resample_prior'].lower()[0] == 'f' or
                        opts['resample_prior'].lower()[0] == '0'
                    ):
                    resample_prior = False
                if 'sampler' in opts:
                    sampler = opts['sampler']
                if 'scale' in opts:
                    try:
                        scale_float = float(opts['scale'])
                        if scale_float > 0. and scale_float < 50.:
                            scale = scale_float
                    except Exception:
                        pass
                if 'seed' in opts:
                    try:
                        seed_int = int(opts['seed'])
                        if seed_int >= 0 and seed_int <= MAX_SEED:
                            seed = seed_int
                    except Exception:
                        pass
                if 'steps' in opts:
                    try:
                        steps_int = int(opts['steps'])
                        if steps_int >= MIN_STEPS and steps_int <= MAX_STEPS:
                            steps = steps_int
                    except Exception:
                        pass
                if 'strength' in opts:
                    try:
                        strength_float = float(opts['strength'])
                        if strength_float >= 0.5 and strength_float <= 1.:
                            strength = strength_float
                    except Exception:
                        pass
                if 'width' in opts:
                    try:
                        width_int = int(opts['width'])
                        if width_int in VALID_IMAGE_HEIGHT_WIDTH:
                            width = width_int
                    except Exception:
                        pass
            prompt = prompt[0:parens_idx]

        await actions.interpolate(message.channel, message.author, client,
            prompt.split('|')[0], prompt.split('|')[1],
            height=height,
            resample_prior=resample_prior,
            sampler=sampler,
            scale=scale,
            seed=seed,
            steps=steps,
            strength=strength,
            width=width)
        return
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>upscale '):
        msg_split = message.clean_content.split(' ')
        if len(msg_split) < 3:
            await message.channel.send('Upscale requires at least two arguments')
            return

        docarray_id = msg_split[1]
        idx = 0
        try:
            idx = int(msg_split[2])
        except Exception:
            pass

        upscaler = None
        if len(msg_split) > 3:
            parens_idx = msg_split[3].find('(')
            if parens_idx >= 0:
                text = msg_split[3][parens_idx:]
                if len(text) > 0 and text[0] == '(' and text[-1] == ')':
                    opts = {}
                    try:
                        opts = { val.split('=')[0].strip(): val.split('=')[1].strip()
                            for val in text[1:-1].split(',') }
                    except IndexError:
                        pass

                    if 'upscaler' in opts and opts['upscaler'] in [
                        UPSCALER_NONE,
                        UPSCALER_REALESRGAN_4X,
                        UPSCALER_REALESRGAN_4X_ANIME,
                        UPSCALER_REALESRGAN_4X_FACE,
                        UPSCALER_SWINIR,
                    ]:
                        upscaler = opts['upscaler']

        await actions.upscale(message.channel, message.author, client,
            docarray_id, idx, upscaler=upscaler)
        return
    elif len(message.mentions) == 1 and \
        message.mentions[0].id == client.user.id and \
        len(message.attachments) >= 1:
        for i, attachment in enumerate(message.attachments):
            sid = short_id_generator()
            image_fn = IMAGE_LOCATION_FN(sid)
            da_fn = DOCARRAY_LOCATION_FN(sid)

            image_bytes = await attachment.read()
            try:
                image = Image.open(BytesIO(image_bytes))
            except Exception:
                await message.channel.send(f'Could not load image file for attachment {i}')
                continue

            image = resize_image(image)
            image.save(image_fn, format='PNG')

            buffered = BytesIO()
            image.save(buffered, format='PNG')
            _d = Document(
                blob=buffered.getvalue(),
                mime_type='image/png',
                tags={
                    'text': '',
                    'generator': 'discord image upload',
                    'request_time': time.time(),
                },
            ).convert_blob_to_datauri()

            clean_content = ''
            clean_content_split = message.clean_content.split(' ')
            if len(clean_content_split) > 1:
                clean_content = ' '.join(clean_content_split)
            _d.text = clean_content

            da = DocumentArray([_d])
            da.save_binary(da_fn, protocol='protobuf', compress='lz4')
            await message.channel.send(f'Attachment {attachment.filename} ({i}) sent by ' +
                f'<@{str(message.author.id)}> has been uploaded and given ID `{sid}`.' +
                ' To use this ID in a riff or upscale, just use 0 for the image ' +
                'index.')
            await message.channel.send(sid)


@client.event
async def on_ready():
    print('Loading old buttons back into memory')
    now = int(time.time())

    # Default is two hours to look back ewhen loading.
    reload_last_seconds = 120 * 60
    if args.reload_last_minutes is not None:
        reload_last_seconds = args.reload_last_minutes * 60

    # init the button handler and load up any previously saved buttons. Skip
    # any buttons that are more than reload_last_seconds old.
    for view_dict in tqdm(button_store_dict[BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY]):
        if view_dict['time'] >= now - reload_last_seconds:
            try:
                view = FourImageButtons.from_serialized(client, view_dict)
            except KeyError:
                continue
            client.add_view(view, message_id=view_dict['message_id'])

    print('Bot is alive')


client.run(args.token)
