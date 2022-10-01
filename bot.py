import argparse
import asyncio
import datetime
import json
import os
import pathlib
import random
import re
import string
import sys
import time

from io import BytesIO
from typing import Optional, Union
from urllib.error import URLError
from urllib.request import urlopen

import discord
import numpy as np

from PIL import Image
from discord import app_commands
from docarray import Document, DocumentArray
from tqdm import tqdm
from transformers import CLIPTokenizer

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SELF_DIR))

from serializers import (
    serialize_image_request,
    serialize_interpolate_request,
    serialize_riff_request,
    serialize_upscale_request,
)


parser = argparse.ArgumentParser()
parser.add_argument('token', help='Discord token')
parser.add_argument('--allow-queue', dest='allow_queue',
    action=argparse.BooleanOptionalAction)
parser.add_argument('--default-steps', dest='default_steps', nargs='?',
    type=int, help='Default number of steps for the sampler', default=50)
parser.add_argument('--hours-on-server-to-use', dest='hours_needed', nargs='?',
    type=int,
    help='The hours the user has been on the server before they can use the bot',
    required=False)
parser.add_argument('-g', '--guild', dest='guild',
    help='Discord guild ID', type=int, required=False)
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
nsfw_toxic_detection_fn = None
nsfw_wordlist: list[str] = []
safety_feature_extractor = None
safety_checker = None
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

guild = args.guild

# In memory k-v stores.
currently_fetching_ai_image: dict[str, Union[str, bool]] = {}
user_image_generation_nonces: dict[str, int] = {}

BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY = 'four_img_views'
button_store_dict: dict[str, list] = { BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY: [] }

MANUAL_LINK = 'https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/tree/master/manual#readme'
SD_CONCEPTS_URL_FN = lambda concept: f'https://huggingface.co/sd-concepts-library/{concept}/resolve/main/'

REGEX_FOR_ID = re.compile('([0-9a-zA-Z]){12}$')
REGEX_FOR_TAGS = re.compile('<.*?>')

ID_LENGTH = 12
BUTTON_STORE = f'temp_json/button-store-{str(guild)}.json'
CLIP_TOKENIZER_MERGES_FN = 'clip_vit_large_patch14/merges.txt'
CLIP_TOKENIZER_VOCAB_FN = 'clip_vit_large_patch14/vocab.json'
DISCORD_EMBED_MAX_LENGTH = 1024
DOCARRAY_LOCATION_FN = lambda docarray_id: f'image_docarrays/{docarray_id}.bin'
IMAGE_LOCATION_FN = lambda sid: f'images/{sid}.png'
JSON_IMAGE_TOOL_INPUT_FILE_FN = lambda uid, nonce: f'temp_json/request-{uid}_{nonce}.json'
JSON_IMAGE_TOOL_OUTPUT_FILE_FN = lambda uid, nonce: f'temp_json/output-{uid}_{nonce}.json'
MIN_ITERATIONS = 1
MAX_ITERATIONS = 16
MIN_SCALE = 1.0
MAX_SCALE = 50.0
MAX_SEED = 2 ** 32 - 1
MIN_STEPS = 10
MAX_STEPS = 250
MIN_STRENGTH = 0.01
MIN_STRENGTH_INTERPOLATE = 0.50
MAX_STRENGTH = 0.99
MIN_IMAGE_HEIGHT_WIDTH = 384
MAX_IMAGE_HEIGHT_WIDTH = 768
MAX_MODEL_CLIP_TOKENS_PER_PROMPT = 77
NUM_IMAGES_MAX = 9
VALID_IMAGE_HEIGHT_WIDTH = { 384, 416, 448, 480, 512, 544, 576, 608, 640, 672,
    704, 736, 768 }
VALID_TAG_CONCEPTS = {}


def short_id_generator():
    return ''.join(random.choices(string.ascii_lowercase +
        string.ascii_uppercase + string.digits, k=ID_LENGTH))


SAMPLER_CHOICES = [
    app_commands.Choice(name="k_lms", value="k_lms"),
    app_commands.Choice(name="dpm2", value="dpm2"),
    app_commands.Choice(name="dpm2_ancestral", value="dpm2_ancestral"),
    app_commands.Choice(name="heun", value="heun"),
    app_commands.Choice(name="euler", value="euler"),
    app_commands.Choice(name="euler_ancestral", value="euler_ancestral"),
]

HEIGHT_AND_WIDTH_CHOICES = [
    app_commands.Choice(name="512 (Default)", value=512),
    app_commands.Choice(name="384", value=384),
    app_commands.Choice(name="416", value=416),
    app_commands.Choice(name="448", value=448),
    app_commands.Choice(name="480", value=480),
    app_commands.Choice(name="544", value=544),
    app_commands.Choice(name="576", value=576),
    app_commands.Choice(name="608", value=608),
    app_commands.Choice(name="640", value=640),
    app_commands.Choice(name="672", value=672),
    app_commands.Choice(name="704", value=704),
    app_commands.Choice(name="736", value=736),
    app_commands.Choice(name="768", value=768),
]

pathlib.Path('./image_docarrays').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images').mkdir(parents=True, exist_ok=True)
pathlib.Path('./temp_json').mkdir(parents=True, exist_ok=True)


# A simple JSON store for button views that we write to when making new buttons
# for new calls and which is kept in memory and keeps track of all buttons ever
# added. This allows us to persist buttons between reboots of the bot, power
# outages, etc.
#
# TODO Clear out old buttons on boot, since we ignore anything more than 48
# hours old below anyway.
bs_fn = pathlib.Path(BUTTON_STORE)
if bs_fn.is_file():
    with open(bs_fn, 'r') as bs:
        button_store_dict = json.load(bs)


def write_button_store():
    global button_store_dict
    with open(bs_fn, 'w') as bs:
        json.dump(button_store_dict, bs)


def resize_image(image):
    w, h = image.size
    ratio = float(w) / float(h)
    
    w = MAX_IMAGE_HEIGHT_WIDTH
    h = MAX_IMAGE_HEIGHT_WIDTH
    if ratio > 1:
        w = MAX_IMAGE_HEIGHT_WIDTH
        h = int(float(MAX_IMAGE_HEIGHT_WIDTH) / ratio)
        if h < MIN_IMAGE_HEIGHT_WIDTH:
            h = MIN_IMAGE_HEIGHT_WIDTH
    if ratio < 1:
        w = int(float(MAX_IMAGE_HEIGHT_WIDTH) * ratio)
        if w < MIN_IMAGE_HEIGHT_WIDTH:
            w = MIN_IMAGE_HEIGHT_WIDTH
        h = MAX_IMAGE_HEIGHT_WIDTH
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    return image.resize((w, h), resample=Image.LANCZOS)


def document_to_pil(doc):
    uri_data = urlopen(doc.uri)
    return Image.open(BytesIO(uri_data.read()))


def bump_nonce_and_return(user_id: str):
    global user_image_generation_nonces
    if user_image_generation_nonces.get(user_id, None) is None:
        user_image_generation_nonces[user_id] = 0
    else:
        user_image_generation_nonces[user_id] += 1
    return user_image_generation_nonces[user_id]


def img_to_tensor(img):
    import torch
    w, h = img.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    img = img.resize((w, h), resample=Image.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    return 2.*img - 1.


def check_safety(img_loc):
    try:
        img = Image.open(img_loc).convert("RGB")
        safety_checker_input = safety_feature_extractor(img,
            return_tensors="pt")
        _, has_nsfw_concept = safety_checker(
            images=[img_to_tensor(img)],
            clip_input=safety_checker_input.pixel_values)
    except Exception as e:
        import traceback
        traceback.print_exc()
    return has_nsfw_concept[0]


def prompt_contains_nsfw(prompt):
    if prompt is None:
        return False
    if nsfw_toxic_detection_fn is not None:
        results = nsfw_toxic_detection_fn(prompt)
        # TODO Allow setting these cutoffs?
        if results['sexual_explicit'] > 0.1 or \
            results['obscene'] > 0.5 or \
            results['toxicity'] > 0.8 or \
            results['severe_toxicity'] > 0.5 or \
            results['identity_attack'] > 0.5:
            return True

    if len(nsfw_wordlist) == 0:
        return False
    return any(word in prompt.lower() for word in nsfw_wordlist)


def prompt_has_valid_sd_custom_embeddings(prompt: str|None):
    '''
    Ensure all custom SD embeddings from huggingface are valid by checking
    their website. Cache the requests too.
    '''
    global VALID_TAG_CONCEPTS

    if type(prompt) != str:
        return

    for tag in re.findall(REGEX_FOR_TAGS, prompt):
        concept = tag[1:-1]
        if VALID_TAG_CONCEPTS.get(concept, False):
            continue
        urlopen(SD_CONCEPTS_URL_FN(concept) + 'token_identifier.txt')
        VALID_TAG_CONCEPTS[concept] = True


def seed_from_docarray_id(docarray_id):
    '''
    Retrieve a seed from a docarray ID and return it. If it is unset return
    None.
    '''
    docarray_loc = DOCARRAY_LOCATION_FN(docarray_id)
    da = DocumentArray.load_binary(
        docarray_loc, protocol='protobuf', compress='lz4'
    )
    original_request = da[0].tags.get('request', None)
    if original_request is not None:
        return int(original_request['seed'])
    return None


def to_discord_file_and_maybe_check_safety(img_loc):
    nsfw = False
    if args.auto_spoiler:
        nsfw = check_safety(img_loc)
    return discord.File(img_loc, spoiler=nsfw)


async def check_user_joined_at(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,
):
    if not args.hours_needed:
        return True
    duration = datetime.datetime.utcnow() - user.joined_at.replace(tzinfo=None)
    hours_int = int(duration.total_seconds()) // 60 ** 2
    if duration < datetime.timedelta(hours=args.hours_needed):
        await channel.send('Sorry, you have not been on this server long enough ' +
            f'to use the bot (needed {args.hours_needed} hours, have ' +
            f'{hours_int} hours).')
        return False
    return True


async def check_subprompt_token_length(
    channel: discord.abc.GuildChannel,
    user_id: str,
    prompt: str,
):
    if prompt is None or prompt == '':
        return True

    prompt_parser = re.compile("""
        (?P<prompt>     # capture group for 'prompt'
        (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
        )               # end 'prompt'
        (?:             # non-capture group
        :+              # match one or more ':' characters
        (?P<weight>     # capture group for 'weight'
        -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
        )?              # end weight capture group, make optional
        \s*             # strip spaces after weight
        |               # OR
        $               # else, if no ':' then match end of line
        )               # end non-capture group
        """, re.VERBOSE)
    parsed_prompts = [match.group('prompt').replace('\\:', ':')
        for match in re.finditer(prompt_parser, prompt)]

    tokenizer = CLIPTokenizer(CLIP_TOKENIZER_VOCAB_FN, CLIP_TOKENIZER_MERGES_FN)
    for subprompt in parsed_prompts:
        if subprompt is None or subprompt == '':
            continue
        as_tokens = tokenizer(subprompt)
        if as_tokens.get('input_ids', None) is None:
            await channel.send(f'Unable to subprompt parse prompt "{subprompt}"')
            return False
        n_tokens = len(as_tokens['input_ids'])
        if n_tokens > MAX_MODEL_CLIP_TOKENS_PER_PROMPT:
            await channel.send(f'Subprompt "{subprompt}" for user <@{user_id}> ' +
                f'is too long (got {n_tokens}, max ' +
                f'{MAX_MODEL_CLIP_TOKENS_PER_PROMPT}). Shorten this subprompt ' +
                'or break into multiple weighted subprompts.')
            return False
    return True


intents = discord.Intents(
    messages=True,
    dm_messages=True,
    guild_messages=True,
    message_content=True,
)


class YASDClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        guild_id = None
        if guild is not None:
            guild_id =  discord.Object(id=guild)
        if guild_id is not None:
            self.tree.copy_global_to(guild=guild_id)
            await self.tree.sync(guild=guild_id)


client = YASDClient()

class FourImageButtons(discord.ui.View):
    RIFF_ASPECT_RATIO_PLACEHOLDER_MESSAGE = 'Select Riff Aspect Ratio'
    RIFF_STRENGTH_PLACEHOLDER_MESSAGE = 'Select Riff Strength (no effect on outriff)'

    pixels_height = 512
    idx_parent = None
    message_id = None
    short_id = None
    short_id_parent = None
    strength = None
    pixels_width = 512
    def __init__(
        self,
        *,
        idx_parent: str|None=None,
        message_id: int|None=None,
        short_id: str|None=None,
        short_id_parent: str|None=None,
        strength: float|None=None,
        timeout=None,
    ):
        super().__init__(timeout=timeout)
        self.idx_parent = idx_parent
        if self.idx_parent is not None and type(self.idx_parent) == float:
            self.idx_parent = int(self.idx_parent)
        self.message_id = message_id
        self.short_id = short_id
        self.short_id_parent = short_id_parent
        self.strength = strength

        self.pixels_width, self.pixels_height = self.original_image_sizes()

    def original_image_sizes(self):
        old_docarray_loc = DOCARRAY_LOCATION_FN(self.short_id)
        da = DocumentArray.load_binary(
            old_docarray_loc, protocol='protobuf', compress='lz4'
        )
        loaded = document_to_pil(da[0])
        return loaded.size

    def serialize_to_json_and_store(self):
        '''
        Store a serialized representation in the global magic json.
        '''
        global button_store_dict
        as_dict = {
            'height': self.pixels_height,
            'idx_parent': self.idx_parent,
            'message_id': self.message_id,
            'short_id': self.short_id,
            'short_id_parent': self.short_id_parent,
            'strength': self.strength,
            'items': [ {
                'label': item.label if getattr(item, 'label', None) is not None
                    else getattr(item, 'placeholder', None),
                'custom_id': item.custom_id,
                'row': item.row,
            } for item in self.children ],
            'time': int(time.time()),
            'width': self.pixels_width,
        }
        button_store_dict[BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY].append(as_dict)
        write_button_store()

    @classmethod
    def from_serialized(cls, serialized):
        '''
        Return a view from a serialized representation.
        '''
        idx_parent = serialized.get('idx_parent', None)
        message_id = serialized['message_id']
        short_id = serialized['short_id']
        short_id_parent = serialized.get('short_id_parent', None)
        strength = serialized.get('strength', None)
        fib = cls(
            idx_parent=idx_parent,
            message_id=message_id,
            short_id=short_id,
            short_id_parent=short_id_parent,
            strength=strength,
        )

        def labels_for_map(item):
            if isinstance(item, discord.ui.Button):
                return item.label
            if isinstance(item, discord.ui.Select):
                return item.placeholder
            return None

        mapped_to_label = { labels_for_map(item): item
            for item in fib.children }
        for item_dict in serialized['items']:
            if item_dict['label'] == fib.RIFF_ASPECT_RATIO_PLACEHOLDER_MESSAGE or \
                item_dict['label'] == fib.RIFF_STRENGTH_PLACEHOLDER_MESSAGE:
                sel = mapped_to_label[item_dict['label']]
                sel.custom_id = item_dict['custom_id']
            else:
                btn = mapped_to_label[item_dict['label']]
                btn.custom_id = item_dict['custom_id']
        
        if serialized.get('height', None) is not None:
            fib.height = int(serialized['height'])
        if serialized.get('width', None) is not None:
            fib.width = int(serialized['width'])

        return fib

    async def global_shows_in_use(self, interaction: discord.Interaction):
        if args.allow_queue: return False

        global currently_fetching_ai_image
        author_id = str(interaction.user.id)
        if not args.allow_queue and currently_fetching_ai_image.get(author_id, False):
            await interaction.channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
            await interaction.response.defer()
            return True
        return False

    async def handle_riff(self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
        idx: int,
    ):
        docarray_loc = DOCARRAY_LOCATION_FN(self.short_id)
        da = DocumentArray.load_binary(
            docarray_loc, protocol='protobuf', compress='lz4'
        )

        latentless = False
        resize = False
        sampler = None
        scale = None
        steps = None
        strength = None

        original_request = da[0].tags.get('request', None)
        if original_request is not None and \
            original_request['api'] == 'txt2img':
            sampler = original_request['sampler']
            scale = original_request['scale']
            steps = int(original_request['steps'])
        if original_request is not None and \
            original_request['api'] == 'stablediffuse':
            latentless = original_request['latentless']
            resize = original_request.get('resize', False)
            sampler = original_request['sampler']
            scale = original_request['scale']
            steps = int(original_request['steps'])
            strength = original_request['strength']

        if self.strength is not None:
            strength = self.strength

        await interaction.response.defer()
        await _riff(interaction.channel, interaction.user, self.short_id, idx,
            height=self.pixels_height,
            latentless=latentless,
            resize=resize,
            sampler=sampler,
            scale=scale,
            seed=random.randint(1, 2 ** 32 - 1),
            steps=steps,
            strength=strength,
            width=self.pixels_width)

    async def handle_retry(self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ):
        await interaction.response.defer()

        docarray_loc = DOCARRAY_LOCATION_FN(self.short_id)
        da = DocumentArray.load_binary(
            docarray_loc, protocol='protobuf', compress='lz4'
        )
        original_request = da[0].tags.get('request', None)
        if original_request is None:
            await interaction.channel.send('No original request could be found')

        width, height = self.original_image_sizes()

        if original_request['api'] == 'txt2img':
            prompt = da[0].tags['text']
            sampler = original_request['sampler']
            scale = original_request['scale']
            steps = int(original_request['steps'])
            await _image(interaction.channel, interaction.user,
                prompt,
                height=height,
                sampler=sampler,
                scale=scale,
                steps=steps,
                width=width)
        if original_request['api'] == 'stablediffuse':
            resize = original_request.get('resize', False)
            latentless = original_request['latentless']
            sampler = original_request['sampler']
            scale = original_request['scale']
            steps = int(original_request['steps'])
            strength = original_request['strength']

            await _riff(
                interaction.channel, interaction.user,
                self.short_id_parent, self.idx_parent,
                height=height,
                latentless=latentless,
                resize=resize,
                sampler=sampler,
                scale=scale,
                seed=random.randint(1, 2 ** 32 - 1),
                steps=steps,
                strength=strength,
                width=width)

    async def handle_upscale(self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
        idx: int,
    ):
        await interaction.response.defer()
        completed = await _upscale(interaction.channel, interaction.user,
            self.short_id, idx)

        if completed:
            button.disabled = True
            await interaction.message.edit(view=self)

    @discord.ui.button(label="Riff 0", style=discord.ButtonStyle.blurple, row=0,
        custom_id=f'{short_id_generator()}-riff-0')
    async def riff_button_0(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_riff(interaction, button, 0)

    @discord.ui.button(label="Riff 1", style=discord.ButtonStyle.blurple, row=0,
        custom_id=f'{short_id_generator()}-riff-1')
    async def riff_button_1(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_riff(interaction, button, 1)

    @discord.ui.button(label="Riff 2", style=discord.ButtonStyle.blurple, row=0,
        custom_id=f'{short_id_generator()}-riff-2')
    async def riff_button_2(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_riff(interaction, button, 2)

    @discord.ui.button(label="Riff 3", style=discord.ButtonStyle.blurple, row=0,
        custom_id=f'{short_id_generator()}-riff-3')
    async def riff_button_3(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_riff(interaction, button, 3)

    @discord.ui.button(label="Retry", style=discord.ButtonStyle.secondary, row=0,
        custom_id=f'{short_id_generator()}-riff-3')
    async def retry_button(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_retry(interaction, button)

    @discord.ui.button(label="Upscale 0", style=discord.ButtonStyle.green, row=1,
        custom_id=f'{short_id_generator()}-upscale-0')
    async def upscale_button_0(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_upscale(interaction, button, 0)

    @discord.ui.button(label="Upscale 1", style=discord.ButtonStyle.green, row=1,
        custom_id=f'{short_id_generator()}-upscale-1')
    async def upscale_button_1(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_upscale(interaction, button, 1)

    @discord.ui.button(label="Upscale 2", style=discord.ButtonStyle.green, row=1,
        custom_id=f'{short_id_generator()}-upscale-2')
    async def upscale_button_2(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_upscale(interaction, button, 2)

    @discord.ui.button(label="Upscale 3", style=discord.ButtonStyle.green, row=1,
        custom_id=f'{short_id_generator()}-upscale-3')
    async def upscale_button_3(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_upscale(interaction, button, 3)

    @discord.ui.select(placeholder=RIFF_ASPECT_RATIO_PLACEHOLDER_MESSAGE, row=2,
        custom_id=f'{short_id_generator()}-riff-select-aspect-ratio',
        options=[
            discord.SelectOption(label='2:1 (landscape)', value='2:1'),
            discord.SelectOption(label='3:2', value='3:2'),
            discord.SelectOption(label='4:3', value='4:3'),
            discord.SelectOption(label='1:1 (square)', value='1:1'),
            discord.SelectOption(label='3:4', value='3:4'),
            discord.SelectOption(label='2:3', value='2:3'),
            discord.SelectOption(label='1:2 (portait)', value='1:2'),
            discord.SelectOption(label='Original Image Size', value='original'),
        ])
    async def select_aspect_ratio(self, interaction: discord.Interaction,
        selection: discord.ui.Select):
        selected = selection.values
        sel = selected[0]
        if sel == '2:1':
            self.pixels_height = 384
            self.pixels_width = 768
        if sel == '3:2': # ish
            self.pixels_height = 448
            self.pixels_width = 704
        if sel == '4:3':
            self.pixels_height = 480
            self.pixels_width = 640
        if sel == '1:1':
            self.pixels_height = 512
            self.pixels_width = 512
        if sel == '3:4':
            self.pixels_height = 640
            self.pixels_width = 480
        if sel == '2:3': # ish
            self.pixels_height = 704
            self.pixels_width = 448
        if sel == '1:2':
            self.pixels_height = 768
            self.pixels_width = 384
        if sel == 'original':
            self.pixels_width, self.pixels_height = self.original_image_sizes()
        await interaction.response.defer()

    @discord.ui.select(placeholder=RIFF_STRENGTH_PLACEHOLDER_MESSAGE, row=3,
        custom_id=f'{short_id_generator()}-riff-select-strength',
        options=[
            discord.SelectOption(label='0.75 (default)', value='0.75'),
            discord.SelectOption(label='0.1', value='0.1'),
            discord.SelectOption(label='0.2', value='0.2'),
            discord.SelectOption(label='0.3', value='0.3'),
            discord.SelectOption(label='0.4', value='0.4'),
            discord.SelectOption(label='0.5', value='0.5'),
            discord.SelectOption(label='0.6', value='0.6'),
            discord.SelectOption(label='0.7', value='0.7'),
            discord.SelectOption(label='0.8', value='0.8'),
            discord.SelectOption(label='0.9', value='0.9'),
        ])
    async def select_strength(self, interaction: discord.Interaction,
        selection: discord.ui.Select):
        selected = selection.values
        if selected[0] is None:
            self.strength = None
        else:
            self.strength = float(selected[0])
        
        await interaction.response.defer()


async def send_alert_embed(
    channel: discord.abc.GuildChannel,
    author_id: str,
    work_msg: discord.Message,
    serialized_cmd: str,
):
    guild_id = None
    if channel.guild is not None:
        guild_id = str(channel.guild.id)
    channel_id = str(channel.id)
    completed_id = str(work_msg.id)
    embed = discord.Embed()
    if guild_id is not None:
        embed.description = f'Your request has finished. [Please view it here](https://discord.com/channels/{guild_id}/{channel_id}/{completed_id}).'
    else:
        embed.description = f'Your request has finished. [Please view it here](https://discord.com/channels/@me/{channel_id}/{completed_id}).'
    serialized_chunks = [
        serialized_cmd[_i:_i + DISCORD_EMBED_MAX_LENGTH]
        for _i in range(0, len(serialized_cmd), DISCORD_EMBED_MAX_LENGTH)
    ]
    if len(serialized_chunks) == 1:
        embed.add_field(name="Command Executed", value=serialized_cmd, inline=False)
    else:
        for idx, chunk in enumerate(serialized_chunks):
            embed.add_field(
                name="Command Executed" if not idx else '',
                value=chunk, inline=False)

    embed.set_thumbnail(url=work_msg.attachments[0].url)
    await channel.send(f'Job completed for <@{author_id}>.', embed=embed)


async def _image(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,

    prompt: str,

    height: Optional[int]=None,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    seed_search: bool=None,
    steps: Optional[int]=args.default_steps,
    width: Optional[int]=None,
):
    global currently_fetching_ai_image
    author_id = str(user.id)

    if args.restrict_all_to_channel:
        if channel.id != args.restrict_all_to_channel:
            await channel.send('You are not allowed to use this in this channel!')
            return

    if steps is None:
        steps = args.default_steps

    short_id = None
    typ = 'prompt'
    if prompt.find('[') != -1 and prompt.find(']') != -1:
        typ = 'promptarray'
    if seed_search:
        typ = 'promptsearch'

    if (args.nsfw_wordlist or nsfw_toxic_detection_fn is not None) and \
        prompt_contains_nsfw(prompt):
        await channel.send('Sorry, this prompt potentially contains NSFW ' +
            'or offensive content.')
        return

    try:
        prompt_has_valid_sd_custom_embeddings(prompt)
    except Exception as e:
        await channel.send('Sorry, one of your custom embeddings is invalid.')
        return

    if not await check_subprompt_token_length(channel, author_id, prompt):
        return

    if not await check_user_joined_at(channel, user):
        return

    if not args.allow_queue and currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return
    currently_fetching_ai_image[author_id] = prompt
    work_msg = await channel.send(
        f'Now beginning work on "{prompt}" for <@{author_id}>. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'height': height,
            'prompt': prompt,
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
            'steps': steps,
            'type': typ,
            'width': width,
        }
        nonce = bump_nonce_and_return(author_id)
        with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id, nonce), 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py', f'{author_id}_{nonce}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id, nonce), 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']
        seeds = output.get('seeds', None)

        file = to_discord_file_and_maybe_check_safety(image_loc)
        if seed_search is True:
            seed_lst = []
            for i, _s in enumerate(seeds):
                seed_lst.append(f'{i}: {_s}')
            seeds_str = ', '.join(seed_lst)

            work_msg = await work_msg.edit(
                content=f'Image generation for prompt "{prompt}" by <@{author_id}> complete. The ID for your images is `{short_id}`. Seeds used were {seeds_str}',
                attachments=[file])
        elif typ == 'promptarray':
            work_msg = await work_msg.edit(
                content=f'Image generation for prompt array "{prompt}" by <@{author_id}> complete. The ID for your images is `{short_id}`.',
                attachments=[file])
        else:
            btns = FourImageButtons(message_id=work_msg.id, short_id=short_id)
            btns.serialize_to_json_and_store()
            client.add_view(btns, message_id=work_msg.id)
            work_msg = await work_msg.edit(
                content=f'Image generation for prompt "{prompt}" by <@{author_id}> complete. The ID for your images is `{short_id}`.',
                attachments=[file],
                view=btns)

        serialized_cmd = serialize_image_request(
            prompt=prompt,
            height=height,
            sampler=sampler,
            scale=scale,
            seed=seed_from_docarray_id(short_id),
            seed_search=seed_search,
            steps=steps,
            width=width)
        await send_alert_embed(channel, author_id, work_msg, serialized_cmd)

    except Exception as e:
        await channel.send(f'Got unknown error on prompt "{prompt}": {str(e)}')
    finally:
        currently_fetching_ai_image[author_id] = False

    return short_id


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

    sid = await _image(interaction.channel, interaction.user, prompt,
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


async def _riff(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,

    docarray_id: str,
    idx: int,

    height: Optional[int]=None,
    iterations: Optional[int]=None,
    latentless: bool=False,
    prompt: Optional[str]=None,
    resize: bool=False,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    steps: Optional[int]=args.default_steps,
    strength: Optional[float]=None,
    width: Optional[int]=None,
):
    global currently_fetching_ai_image
    author_id = str(user.id)

    if args.restrict_all_to_channel:
        if channel.id != args.restrict_all_to_channel:
            await channel.send('You are not allowed to use this in this channel!')
            return

    if REGEX_FOR_ID.match(docarray_id) is None:
        await channel.send(f'Got invalid docarray ID \'{docarray_id}\'')
        return

    short_id = None
    if not args.allow_queue and currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return

    if (args.nsfw_wordlist or nsfw_toxic_detection_fn is not None) and \
        prompt_contains_nsfw(prompt):
        await channel.send('Sorry, this prompt potentially contains NSFW ' +
            'or offensive content.')
        return

    try:
        prompt_has_valid_sd_custom_embeddings(prompt)
    except Exception as e:
        await channel.send('Sorry, one of your custom embeddings is invalid.')
        return

    if not await check_subprompt_token_length(channel, author_id, prompt):
        return

    if not await check_user_joined_at(channel, user):
        return

    currently_fetching_ai_image[author_id] = f'riffs on previous work `{docarray_id}`, index {str(idx)}'
    work_msg = await channel.send(f'Now beginning work on "riff `{docarray_id}` index {str(idx)}" for <@{author_id}>. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'docarray_id': docarray_id,
            'height': height,
            'index': idx,
            'iterations': iterations,
            'latentless': bool(latentless),
            'prompt': prompt,
            'resize': bool(resize),
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
            'steps': steps,
            'strength': strength,
            'type': 'riff',
            'width': width,
        }
        nonce = bump_nonce_and_return(author_id)
        with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id, nonce), 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py', f'{author_id}_{nonce}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id, nonce), 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']

        file = to_discord_file_and_maybe_check_safety(image_loc)
        btns = FourImageButtons(message_id=work_msg.id, idx_parent=idx,
            short_id=short_id, short_id_parent=docarray_id)
        btns.serialize_to_json_and_store()
        client.add_view(btns, message_id=work_msg.id)
        work_msg = await work_msg.edit(
            content=f'Image generation for riff on `{docarray_id}` index {str(idx)} for <@{author_id}> complete. The ID for your new images is `{short_id}`.',
            attachments=[file],
            view=btns)

        serialized_cmd = serialize_riff_request(
            docarray_id=docarray_id,
            idx=idx,
            height=height,
            iterations=iterations,
            latentless=bool(latentless),
            prompt=prompt,
            resize=bool(resize),
            sampler=sampler,
            scale=scale,
            seed=seed_from_docarray_id(short_id),
            steps=steps,
            strength=strength,
            width=width)
        await send_alert_embed(channel, author_id, work_msg, serialized_cmd)
    except Exception as e:
        await channel.send(f'Got unknown error on riff "{docarray_id}" index {str(idx)}: {str(e)}')
    finally:
        currently_fetching_ai_image[author_id] = False

    return short_id


@client.tree.command(
    description='Create an image from a generated image using its ID and an index',
)
@app_commands.describe(
    docarray_id='The ID for the bot-generated image you want to riff',
    height='Height of the image (default=512)',
    idx='The index of the bot generated image you want to riff',
    iterations='Number of diffusion iterations (1 to 16, default=1)',
    latentless='Do not compute latent embeddings from original image (default=False)',
    prompt='Prompt, which overrides the original prompt for the image (default=None)',
    resize='Resize the image when adjusting width/height instead of attempting outriff (default=False)',
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
async def riff(
    interaction: discord.Interaction,

    docarray_id: str,
    idx: app_commands.Range[int, 0, NUM_IMAGES_MAX-1],

    height: Optional[app_commands.Choice[int]] = None,
    iterations: Optional[app_commands.Range[int, MIN_ITERATIONS, MAX_ITERATIONS]] = None,
    latentless: Optional[bool]=False,
    prompt: Optional[str]=None,
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

    sid = await _riff(interaction.channel, interaction.user, docarray_id, idx,
        height=height.value if height is not None else None,
        iterations=iterations,
        latentless=bool(latentless),
        prompt=prompt,
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
        await interaction.followup.send(f'Failed to parse image!')
        return

    image_fn = IMAGE_LOCATION_FN(short_id)
    da_fn = DOCARRAY_LOCATION_FN(short_id)
    image = resize_image(image).convert('RGB')
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

    sid = await _riff(interaction.channel, interaction.user, short_id, 0,
        height=height.value if height is not None else None,
        iterations=iterations,
        latentless=bool(latentless),
        prompt=prompt,
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


async def _interpolate(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,

    prompt1: str,
    prompt2: str,

    height: Optional[int]=None,
    resample_prior: bool=True,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    steps: Optional[int]=args.default_steps,
    strength: Optional[float]=None,
    width: Optional[int]=None,
):
    global currently_fetching_ai_image
    author_id = str(user.id)
    
    prompt1 = prompt1.strip()
    prompt2 = prompt2.strip()

    if args.restrict_all_to_channel:
        if channel.id != args.restrict_all_to_channel:
            await channel.send('You are not allowed to use this in this channel!')
            return

    if not args.allow_queue and currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return

    if (args.nsfw_wordlist or nsfw_toxic_detection_fn is not None) and (
        prompt_contains_nsfw(prompt1) or prompt_contains_nsfw(prompt2)):
        await channel.send('Sorry, these prompts potentially contain NSFW or ' +
            'offensive content.')
        return

    try:
        prompt_has_valid_sd_custom_embeddings(prompt1)
        prompt_has_valid_sd_custom_embeddings(prompt2)
    except Exception as e:
        await channel.send('Sorry, one of your custom embeddings is invalid.')
        return

    if not await check_subprompt_token_length(channel, author_id, prompt1):
        return

    if not await check_subprompt_token_length(channel, author_id, prompt2):
        return

    if not await check_user_joined_at(channel, user):
        return

    short_id = None
    currently_fetching_ai_image[author_id] = f'interpolate on prompt {prompt1} to {prompt2}'
    work_msg = await channel.send(f'Now beginning work on "interpolate `{prompt1}` to `{prompt2}`" for <@{author_id}>. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'height': height,
            'prompt': f'{prompt1}|{prompt2}',
            'resample_prior': resample_prior,
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
            'steps': steps,
            'strength': strength,
            'type': 'interpolate',
            'width': width,
        }
        nonce = bump_nonce_and_return(author_id)
        with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id, nonce), 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py', f'{author_id}_{nonce}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id, nonce), 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']

        file = to_discord_file_and_maybe_check_safety(image_loc)
        work_msg = await work_msg.edit(
            content=f'Image generation for interpolate on `{prompt1}` to `{prompt2}` for <@{author_id}> complete. The ID for your new images is `{short_id}`.',
            attachments=[file])

        serialized_cmd = serialize_interpolate_request(
            prompt1=prompt1,
            prompt2=prompt2,
            height=height,
            resample_prior=resample_prior,
            sampler=sampler,
            scale=scale,
            seed=seed_from_docarray_id(short_id),
            steps=steps,
            strength=strength,
            width=width)
        await send_alert_embed(channel, author_id, work_msg, serialized_cmd)
    except Exception as e:
        await channel.send(f'Got unknown error on interpolate `{prompt1}` to `{prompt2}`: {str(e)}')
    finally:
        currently_fetching_ai_image[author_id] = False

    return short_id


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

    sid = await _interpolate(interaction.channel, interaction.user, prompt1, prompt2,
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


async def _upscale(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,

    docarray_id: str,
    idx: int,
):
    global currently_fetching_ai_image
    author_id = str(user.id)

    if args.restrict_all_to_channel:
        if channel.id != args.restrict_all_to_channel:
            await channel.send('You are not allowed to use this in this channel!')
            return

    if REGEX_FOR_ID.match(docarray_id) is None:
        await channel.send(f'Got invalid docarray ID \'{docarray_id}\'')
        return

    if not args.allow_queue and currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return

    currently_fetching_ai_image[author_id] = f'upscale on previous work `{docarray_id}`, index {str(idx)}'
    work_msg = await channel.send(f'Now beginning work on "upscale `{docarray_id}` index {str(idx)}" for <@{author_id}>. Please be patient until I finish that.')
    completed = False
    try:
        # Make the request in the filesystem pipeline
        req = {
            'docarray_id': docarray_id,
            'index': idx,
            'type': 'upscale',
        }
        nonce = bump_nonce_and_return(author_id)
        with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id, nonce), 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py', f'{author_id}_{nonce}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id, nonce), 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']

        file = to_discord_file_and_maybe_check_safety(image_loc)
        work_msg = await work_msg.edit(
            content=f'Image generation for upscale on `{docarray_id}` index {str(idx)} for <@{author_id}> complete.',
            attachments=[file])

        serialized_cmd = serialize_upscale_request(docarray_id=docarray_id,
            idx=idx)
        await send_alert_embed(channel, author_id, work_msg, serialized_cmd)
        completed = True
    except Exception as e:
        await channel.send(f'Got unknown error on upscale "{docarray_id}" index {str(idx)}: {str(e)}')
    finally:
        currently_fetching_ai_image[author_id] = False

    return completed


@client.tree.command(
    description='Upscale a generated image using its ID and index',
)
@app_commands.describe(
    docarray_id='The ID for the bot-generated image you want to upscale',
    idx='The index of the bot generated image you want to upscale',
)
async def upscale(
    interaction: discord.Interaction,
    docarray_id: str,
    idx: app_commands.Range[int, 0, NUM_IMAGES_MAX-1],
):
    await interaction.response.defer(thinking=True)

    if args.restrict_slash_to_channel:
        if interaction.channel.id != args.restrict_slash_to_channel:
            await interaction.followup.send('You are not allowed to use this in this channel!')
            return

    await _upscale(interaction.channel, interaction.user, docarray_id, idx)
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
        await _image(message.channel, message.author, prompt, height,
            sampler, scale, seed, seed_search, steps, width)
        return
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>riff '):
        msg_split = message.clean_content.split(' ')
        if len(msg_split) < 3:
            message.channel.send('Riff requires at least two arguments')
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
        prompt = None
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
            if 'prompt' in opts:
                prompt = opts['prompt']
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

        await _riff(message.channel, message.author, docarray_id, idx,
            height=height,
            iterations=iterations,
            latentless=bool(latentless),
            prompt=prompt,
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
                message.channel.send(
                    f'Could not load image file for attachment {message.attachments[0].filename}')
                return

            image = resize_image(image).convert('RGB')
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

        await _riff(message.channel, message.author, sid, 0,
            iterations=iterations,
            latentless=latentless,
            prompt=prompt,
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

        await _interpolate(message.channel, message.author,
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
            message.channel.send('Riff requires at least two arguments')
            return

        docarray_id = msg_split[1]
        idx = 0
        try:
            idx = int(msg_split[2])
        except Exception:
            pass

        await _upscale(message.channel, message.author, docarray_id, idx)
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
                message.channel.send(f'Could not load image file for attachment {i}')
                continue

            image = resize_image(image).convert('RGB')
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
    forty_eight_hours_ago = now - 2 * 24 * 60 * 60
    # init the button handler and load up any previously saved buttons. Skip
    # any buttons that are more than 48 hours old.
    for view_dict in tqdm(button_store_dict[BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY]):
        if view_dict['time'] >= forty_eight_hours_ago:
            view = FourImageButtons.from_serialized(view_dict)
            client.add_view(view, message_id=view_dict['message_id'])

    print('Bot is alive')


client.run(args.token)
