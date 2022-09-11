import argparse
import asyncio
import json
import pathlib
import random
import string
import time

from io import BytesIO
from typing import Any, Optional, Union
from urllib.request import urlopen

import discord

from PIL import Image
from discord import app_commands
from discord.ext import commands
from discord.ui import Button, View
from docarray import Document, DocumentArray


parser = argparse.ArgumentParser()
parser.add_argument('token', help='Discord token')
parser.add_argument('--allow-queue', dest='allow_queue',
    action=argparse.BooleanOptionalAction)
parser.add_argument('--default-steps', dest='default_steps', nargs='?',
    type=int, help='Default number of steps for the sampler', default=50)
parser.add_argument('-g', '--guild', help='Discord guild ID', type=int,
    required=False)
parser.add_argument('--optimized-sd', dest='optimized_sd',
    action=argparse.BooleanOptionalAction)
args = parser.parse_args()

guild = args.guild

# In memory k-v stores.
currently_fetching_ai_image: dict[str, Union[str, bool]] = {}
user_image_generation_nonces: dict[str, int] = {}

BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY = 'four_img_views'
button_store_dict: dict[str, list] = { BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY: [] }

MANUAL_LINK = 'https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/tree/master/manual#readme'
ID_LENGTH = 12
BUTTON_STORE = f'temp_json/button-store-{str(guild)}.json'
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
NUM_IMAGES_MAX = 9
MIN_IMAGE_HEIGHT_WIDTH = 384
MAX_IMAGE_HEIGHT_WIDTH = 768
VALID_IMAGE_HEIGHT_WIDTH = { 384, 448, 512, 576, 640, 704, 768 }


def short_id_generator():
    return ''.join(random.choices(string.ascii_lowercase +
        string.ascii_uppercase + string.digits, k=ID_LENGTH))


if args.optimized_sd:
    SAMPLER_CHOICES = [
        app_commands.Choice(name="ddim", value="ddim"),
    ]
else:
    SAMPLER_CHOICES = [
        app_commands.Choice(name="k_lms", value="k_lms"),
        app_commands.Choice(name="ddim", value="ddim"),
        app_commands.Choice(name="dpm2", value="dpm2"),
        app_commands.Choice(name="dpm2_ancestral", value="dpm2_ancestral"),
        app_commands.Choice(name="heun", value="heun"),
        app_commands.Choice(name="euler", value="euler"),
        app_commands.Choice(name="euler_ancestral", value="euler_ancestral"),
    ]

HEIGHT_AND_WIDTH_CHOICES = [
    app_commands.Choice(name="512", value=512),
    app_commands.Choice(name="384", value=384),
    app_commands.Choice(name="448", value=448),
    app_commands.Choice(name="576", value=576),
    app_commands.Choice(name="640", value=640),
    app_commands.Choice(name="704", value=704),
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
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
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

    pixels_height = 512
    message_id = None
    short_id = None
    pixels_width = 512
    def __init__(self, *, message_id=None, short_id=None, timeout=None):
        super().__init__(timeout=timeout)
        self.message_id = message_id
        self.short_id = short_id

        old_docarray_loc = f'image_docarrays/{short_id}.bin'
        da = DocumentArray.load_binary(
            old_docarray_loc, protocol='protobuf', compress='lz4'
        )
        loaded = document_to_pil(da[0])
        orig_width, orig_height = loaded.size
        self.pixels_width = orig_width
        self.pixels_height = orig_height

    def serialize_to_json_and_store(self):
        '''
        Store a serialized representation in the global magic json.
        '''
        global button_store_dict
        as_dict = {
            'height': self.pixels_height,
            'message_id': self.message_id,
            'short_id': self.short_id,
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
        message_id = serialized['message_id']
        short_id = serialized['short_id']
        fib = cls(message_id=message_id, short_id=short_id)

        def labels_for_map(item):
            if isinstance(item, discord.ui.Button):
                return item.label
            if isinstance(item, discord.ui.Select):
                return item.placeholder
            return None

        mapped_to_label = { labels_for_map(item): item
            for item in fib.children }
        for item_dict in serialized['items']:
            if item_dict['label'] == fib.RIFF_ASPECT_RATIO_PLACEHOLDER_MESSAGE:
                btn = mapped_to_label[item_dict['label']]
                btn.custom_id = item_dict['custom_id']
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
        await interaction.response.defer()
        await _riff(interaction.channel, interaction.user, self.short_id, idx,
            height=self.pixels_height, width=self.pixels_width)

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
            discord.SelectOption(label='1:1 (square)', value='1:1'),
            discord.SelectOption(label='2:3', value='2:3'),
            discord.SelectOption(label='1:2 (portait)', value='1:2'),
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
        if sel == '1:1':
            self.pixels_height = 512
            self.pixels_width = 512
        if sel == '2:3': # ish
            self.pixels_height = 704
            self.pixels_width = 448
        if sel == '1:2':
            self.pixels_height = 768
            self.pixels_width = 384
        await interaction.response.defer()



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

    if steps is None:
        steps = args.default_steps

    short_id = None
    typ = 'prompt'
    if prompt.find('[') != -1 and prompt.find(']') != -1:
        typ = 'promptarray'
    if seed_search:
        typ = 'promptsearch'

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
        file = discord.File(image_loc)
        if seed_search is True:
            await work_msg.edit(
                content=f'Image generation for prompt "{prompt}" by <@{author_id}> complete. The ID for your images is `{short_id}`.',
                attachments=[file])
        else:
            btns = FourImageButtons(message_id=work_msg.id, short_id=short_id)
            btns.serialize_to_json_and_store()
            client.add_view(btns, message_id=work_msg.id)
            await work_msg.edit(
                content=f'Image generation for prompt "{prompt}" by <@{author_id}> complete. The ID for your images is `{short_id}`.',
                attachments=[file],
                view=btns)
        if seed_search is True:
            await channel.send(short_id)
        if seeds is not None:
            seed_lst = []
            for i, _s in enumerate(seeds):
                seed_lst.append(f'{i}: {_s}')
            seeds_str = ', '.join(seed_lst)
            await channel.send(f'Seeds used were {seeds_str}')
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
    height='Height of the image',
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0)',
    seed='Deterministic seed for prompt (1 to 2^32-1)',
    seed_search='Seed searching mode, enumerates 9 different seeds starting at given seed',
    steps='Number of steps to perform (10 to 250)',
    width='Width of the image',
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
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    strength: Optional[float]=None,
    width: Optional[int]=None,
):
    global currently_fetching_ai_image
    author_id = str(user.id)

    short_id = None
    if not args.allow_queue and currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
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
            'latentless': latentless,
            'prompt': prompt,
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
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

        file = discord.File(image_loc)
        btns = FourImageButtons(message_id=work_msg.id, short_id=short_id)
        btns.serialize_to_json_and_store()
        client.add_view(btns, message_id=work_msg.id)
        await work_msg.edit(
            content=f'Image generation for riff on `{docarray_id}` index {str(idx)} for <@{author_id}> complete. The ID for your new images is `{short_id}`.',
            attachments=[file],
            view=btns)
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
    height='Height of the image',
    idx='The index of the bot generated image you want to riff',
    iterations='Number of diffusion iterations (1 to 16)',
    latentless='Do not compute latent embeddings from original image',
    prompt='Prompt, which overrides the original prompt for the image',
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0)',
    seed='Deterministic seed for prompt (1 to 2^32-1)',
    strength="Strength of conditioning (0.01 <= strength <= 0.99)",
    width='Width of the image',
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
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, MIN_SCALE, MAX_SCALE]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    strength: Optional[app_commands.Range[float, MIN_STRENGTH, MAX_STRENGTH]] = None,
    width: Optional[app_commands.Choice[int]] = None,
):
    await interaction.response.defer(thinking=True)
    sid = await _riff(interaction.channel, interaction.user, docarray_id, idx,
        height=height.value if height is not None else None,
        iterations=iterations,
        latentless=bool(latentless),
        prompt=prompt,
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
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
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    strength: Optional[float]=None,
    width: Optional[int]=None,
):
    global currently_fetching_ai_image
    author_id = str(user.id)
    if not args.allow_queue and currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return

    short_id = None
    currently_fetching_ai_image[author_id] = f'interpolate on prompt {prompt1} to {prompt2}'
    work_msg = await channel.send(f'Now beginning work on "interpolate `{prompt1}` to `{prompt2}`" for <@{author_id}>. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'height': height,
            'prompt': f'{prompt1}|{prompt2}',
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
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

        file = discord.File(image_loc)
        await work_msg.edit(
            content=f'Image generation for interpolate on `{prompt1}` to `{prompt2}` for <@{author_id}> complete. The ID for your new images is `{short_id}`.',
            attachments=[file])
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
    height='Height of the image',
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0)',
    seed='Deterministic seed for prompt (1 to 2^32-1)',
    strength="Strength of conditioning (0.01 <= strength <= 0.99)",
    width='Height of the image',
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
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, MIN_SCALE, MAX_SCALE]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    strength: Optional[app_commands.Range[float, MIN_STRENGTH, MAX_STRENGTH]] = None,
    width: Optional[app_commands.Choice[int]] = None,
):
    await interaction.response.defer(thinking=True)
    sid = await _interpolate(interaction.channel, interaction.user, prompt1, prompt2,
        height=height.value if height is not None else None,
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
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

        file = discord.File(image_loc)
        await work_msg.edit(
            content=f'Image generation for upscale on `{docarray_id}` index {str(idx)} for <@{author_id}> complete.',
            attachments=[file])
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
        steps = None
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
        sampler = None
        scale = None
        seed = None
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
            latentless=latentless,
            prompt=prompt,
            sampler=sampler,
            scale=scale,
            seed=seed,
            strength=strength,
            width=width)
        return 
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>image2image'):
        prompt = message.clean_content[13:]
        sid = short_id_generator()
        image_fn = f'images/{sid}.png'
        da_fn = f'image_docarrays/{sid}.bin'
        if len(message.attachments) != 1:
            await message.channel.send(
                'Please upload a single image with your message')
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
        sampler = None
        scale = None
        seed = None
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
            sampler=sampler,
            scale=scale,
            seed=seed,
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
        sampler = None
        scale = None
        seed = None
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
            sampler=sampler,
            scale=scale,
            seed=seed,
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
            image_fn = f'images/{sid}.png'
            da_fn = f'image_docarrays/{sid}.bin'

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
    for view_dict in button_store_dict[BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY]:
        if view_dict['time'] >= forty_eight_hours_ago:
            view = FourImageButtons.from_serialized(view_dict)
            client.add_view(view, message_id=view_dict['message_id'])

    print('Bot is alive')


client.run(args.token)
