import argparse
import asyncio
import json
import pathlib
import time

from docarray import Document, DocumentArray
from io import BytesIO
from typing import Optional, Union

import discord

from PIL import Image
from discord import app_commands
from discord.ext import commands
from shortid import ShortId


parser = argparse.ArgumentParser()
parser.add_argument('token', help='Discord token')
parser.add_argument('-g', '--guild', help='Discord guild ID', type=int,
    required=False)
args = parser.parse_args()
guild = args.guild

currently_fetching_ai_image: dict[str, Union[str, bool]] = {}
short_id_generator = ShortId()

MANUAL_LINK = 'https://github.com/AmericanPresidentJimmyCarter/yasd-discord-bot/tree/master/manual#readme'
JSON_IMAGE_TOOL_INPUT_FILE_FN = lambda uid: f'temp_json/request-{uid}.json'
JSON_IMAGE_TOOL_OUTPUT_FILE_FN = lambda uid: f'temp_json/output-{uid}.json'
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

SAMPLER_CHOICES = [
    app_commands.Choice(name="k_lms", value="k_lms"),
    app_commands.Choice(name="ddim", value="ddim"),
    app_commands.Choice(name="dpm2", value="dpm2"),
    app_commands.Choice(name="dpm2_ancestral", value="dpm2_ancestral"),
    app_commands.Choice(name="heun", value="heun"),
    app_commands.Choice(name="euler", value="euler"),
    app_commands.Choice(name="euler_ancestral", value="euler_ancestral"),
]

pathlib.Path('./image_docarrays').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images').mkdir(parents=True, exist_ok=True)
pathlib.Path('./temp_json').mkdir(parents=True, exist_ok=True)

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
    short_id = None
    def __init__(self, *, short_id=None, timeout=180):
        super().__init__(timeout=timeout)
        self.short_id = short_id

    async def global_shows_in_use(self, interaction: discord.Interaction):
        global currently_fetching_ai_image
        author_id = str(interaction.user.id)
        if currently_fetching_ai_image.get(author_id, False):
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
        await _riff(interaction.channel, interaction.user, self.short_id, idx)

    async def handle_upscale(self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
        idx: int,
    ):
        button.disabled = True
        await interaction.message.edit(view=self)
        await interaction.response.defer()
        await _upscale(interaction.channel, interaction.user, self.short_id, idx)


    @discord.ui.button(label="Riff 0", style=discord.ButtonStyle.blurple)
    async def riff_button_0(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_riff(interaction, button, 0)

    @discord.ui.button(label="Riff 1", style=discord.ButtonStyle.blurple)
    async def riff_button_1(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_riff(interaction, button, 1)

    @discord.ui.button(label="Riff 2", style=discord.ButtonStyle.blurple)
    async def riff_button_2(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_riff(interaction, button, 2)

    @discord.ui.button(label="Riff 3", style=discord.ButtonStyle.blurple) # or .primary
    async def riff_button_3(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_riff(interaction, button, 3)

    @discord.ui.button(label="Upscale 0", style=discord.ButtonStyle.green) # or .primary
    async def upscale_button_0(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_upscale(interaction, button, 0)

    @discord.ui.button(label="Upscale 1", style=discord.ButtonStyle.green) # or .primary
    async def upscale_button_1(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_upscale(interaction, button, 1)

    @discord.ui.button(label="Upscale 2", style=discord.ButtonStyle.green) # or .primary
    async def upscale_button_2(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_upscale(interaction, button, 2)

    @discord.ui.button(label="Upscale 3", style=discord.ButtonStyle.green) # or .primary
    async def upscale_button_3(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        inuse = await self.global_shows_in_use(interaction)
        if inuse:
            return
        await self.handle_upscale(interaction, button, 3)


async def _image(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,

    prompt: str,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    seed_search: bool=None,
    steps: Optional[int]=None,
):
    global currently_fetching_ai_image
    author_id = str(user.id)

    short_id = None
    typ = 'prompt'
    if prompt.find('[') != -1 and prompt.find(']') != -1:
        typ = 'promptarray'
    if seed_search:
        typ = 'promptsearch'

    if currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return
    currently_fetching_ai_image[author_id] = prompt
    work_msg = await channel.send(
        f'Now beginning work on "{prompt}" for {user.display_name}. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'prompt': prompt,
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
            'steps': steps,
            'type': typ,
        }
        with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id), 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py', author_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id), 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']
        seeds = output.get('seeds', None)
        file = discord.File(image_loc)
        await work_msg.edit(
            content=f'Image generation for prompt "{prompt}" complete. The ID for your images is `{short_id}`.',
            attachments=[file],
            view=FourImageButtons(short_id=short_id))
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
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default k_lms)',
    scale='conditioning scale for prompt (1.0 to 50.0)',
    seed='deterministic seed for prompt (1 to 2^32-1)',
    seed_search='seed searching mode, enumerates 9 different seeds starting at given seed',
    steps='number of steps to perform (10 to 250)',
)
@app_commands.choices(sampler=SAMPLER_CHOICES)
async def image(
    interaction: discord.Interaction,

    prompt: str,
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, 1.0, 50.0]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    seed_search: Optional[bool]=False,
    steps: Optional[app_commands.Range[int, MIN_STEPS, MAX_STEPS]] = None,
):
    await interaction.response.defer(thinking=True)
    sid = await _image(interaction.channel, interaction.user, prompt,
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
        seed_search=bool(seed_search),
        steps=steps)
    if sid is not None:
        await interaction.followup.send(sid)
    else:
        await interaction.followup.send('Failed!')


async def _riff(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,

    docarray_id: str,
    idx: int,
    iterations: Optional[int]=None,
    latentless: bool=False,
    prompt: Optional[str]=None,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    strength: Optional[float]=None,
):
    global currently_fetching_ai_image
    author_id = str(user.id)

    short_id = None
    if currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return
    currently_fetching_ai_image[author_id] = f'riffs on previous work `{docarray_id}`, index {str(idx)}'
    work_msg = await channel.send(f'Now beginning work on "riff `{docarray_id}` index {str(idx)}" for {user.display_name}. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'docarray_id': docarray_id,
            'index': idx,
            'iterations': iterations,
            'latentless': latentless,
            'prompt': prompt,
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
            'strength': strength,
            'type': 'riff',
        }
        with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id), 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py', author_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id), 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']

        file = discord.File(image_loc)
        await work_msg.edit(
            content=f'Image generation for riff on `{docarray_id}` index {str(idx)} complete. The ID for your new images is `{short_id}`.',
            attachments=[file],
            view=FourImageButtons(short_id=short_id))
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
    idx='The index of the bot generated image you want to riff',
    iterations='Number of diffusion iterations (1 to 16)',
    latentless='Do not compute latent embeddings from original image',
    prompt='Prompt, which overrides the original prompt for the image',
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0)',
    seed='Deterministic seed for prompt (1 to 2^32-1)',
    strength="Strength of conditioning (0.01 <= strength <= 0.99)",
)
@app_commands.choices(sampler=SAMPLER_CHOICES)
async def riff(
    interaction: discord.Interaction,
    docarray_id: str,
    idx: app_commands.Range[int, 0, NUM_IMAGES_MAX-1],
    iterations: Optional[app_commands.Range[int, MIN_ITERATIONS, MAX_ITERATIONS]] = None,
    latentless: Optional[bool]=False,
    prompt: Optional[str]=None,
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, MIN_SCALE, MAX_SCALE]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    strength: Optional[app_commands.Range[float, MIN_STRENGTH, MAX_STRENGTH]] = None,
):
    await interaction.response.defer(thinking=True)
    sid = await _riff(interaction.channel, interaction.user, docarray_id, idx,
        iterations=iterations,
        latentless=bool(latentless),
        prompt=prompt,
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
        strength=strength)
    if sid is not None:
        await interaction.followup.send(sid)
    else:
        await interaction.followup.send('Failed!')


async def _interpolate(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,

    prompt1: str,
    prompt2: str,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    strength: Optional[float]=None,
):
    global currently_fetching_ai_image
    author_id = str(user.id)
    if currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return

    short_id = None
    currently_fetching_ai_image[author_id] = f'interpolate on prompt {prompt1} to {prompt2}'
    work_msg = await channel.send(f'Now beginning work on "interpolate `{prompt1}` to `{prompt2}`" for {user.display_name}. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'type': 'interpolate',
            'prompt': f'{prompt1}|{prompt2}',
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
            'strength': strength,
        }
        with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id), 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py', author_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id), 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']

        file = discord.File(image_loc)
        await work_msg.edit(
            content=f'Image generation for interpolate on `{prompt1}` to `{prompt2}` complete. The ID for your new images is `{short_id}`.',
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
    sampler='Which sampling algorithm to use (k_lms, ddim, dpm2, dpm2_ancestral, heun, euler, or euler_ancestral. default k_lms)',
    scale='Conditioning scale for prompt (1.0 to 50.0)',
    seed='Deterministic seed for prompt (1 to 2^32-1)',
    strength="Strength of conditioning (0.01 <= strength <= 0.99)",
)
@app_commands.choices(sampler=SAMPLER_CHOICES)
async def interpolate(
    interaction: discord.Interaction,
    prompt1: str,
    prompt2: str,
    sampler: Optional[app_commands.Choice[str]] = None,
    scale: Optional[app_commands.Range[float, MIN_SCALE, MAX_SCALE]] = None,
    seed: Optional[app_commands.Range[int, 0, MAX_SEED]] = None,
    strength: Optional[app_commands.Range[float, MIN_STRENGTH, MAX_STRENGTH]] = None,
):
    await interaction.response.defer(thinking=True)
    sid = await _interpolate(interaction.channel, interaction.user, prompt1, prompt2,
        sampler=sampler.value if sampler is not None else None,
        scale=scale,
        seed=seed,
        strength=strength)
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
    if currently_fetching_ai_image.get(author_id, False) is not False:
        await channel.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image[author_id]}". Please be patient until I finish that.',
            delete_after=5)
        return

    currently_fetching_ai_image[author_id] = f'upscale on previous work `{docarray_id}`, index {str(idx)}'
    work_msg = await channel.send(f'Now beginning work on "upscale `{docarray_id}` index {str(idx)}" for {user.display_name}. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'docarray_id': docarray_id,
            'index': idx,
            'type': 'upscale',
        }
        with open(JSON_IMAGE_TOOL_INPUT_FILE_FN(author_id), 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py', author_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE_FN(author_id), 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']

        file = discord.File(image_loc)
        await work_msg.edit(
            content=f'Image generation for upscale on `{docarray_id}` index {str(idx)} complete.',
            attachments=[file])
    except Exception as e:
        await channel.send(f'Got unknown error on upscale "{docarray_id}" index {str(idx)}: {str(e)}')
    finally:
        currently_fetching_ai_image[author_id] = False


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
    images to store and use to generate riffs as the old image2image pipeline.
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
        sampler = None
        scale = None
        seed = None
        seed_search = False
        steps = None

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
            prompt = prompt[0:parens_idx]
        await _image(message.channel, message.author, prompt,
            sampler, scale, seed, seed_search, steps)
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
            int(msg_split[2])
        except Exception:
            pass

        text = ''
        if len(msg_split) > 3:
            text = ' '.join(msg_split[3:])

        iterations = None
        latentless = False
        prompt = None
        sampler = None
        scale = None
        seed = None
        strength = None
        if len(text) > 0 and text[0] == '(' and text[-1] == ')':
            opts = {}
            try:
                opts = { val.split('=')[0].strip(): val.split('=')[1].strip()
                    for val in text[1:-1].split(',') }
            except IndexError:
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

        await _riff(message.channel, message.author, docarray_id, idx,
            iterations=iterations,
            latentless=latentless,
            prompt=prompt,
            sampler=sampler,
            scale=scale,
            seed=seed,
            strength=strength)
        return 
    if isinstance(message.clean_content, str) and \
        message.clean_content.startswith('>image2image'):
        prompt = message.clean_content[13:]
        sid = short_id_generator.generate()
        image_fn = f'images/{sid}.png'
        da_fn = f'image_docarrays/{sid}.bin'
        if len(message.attachments) != 1:
            await message.channel.send(
                'Please upload a single image with your message (square is best)')
        else:
            image_bytes = await message.attachments[0].read()
            try:
                image = Image.open(BytesIO(image_bytes))
            except Exception:
                message.channel.send(
                    f'Could not load image file for attachment {message.attachments[0].filename}')
                return

            # TODO Allow non-square image sizes
            image = image.resize((512, 512)).convert('RGB')
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

        await _interpolate(message.channel, message.author,
            prompt.split('|')[0], prompt.split('|')[1],
            sampler=sampler,
            scale=scale,
            seed=seed,
            strength=strength)
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
            int(msg_split[2])
        except Exception:
            pass

        await _upscale(message.channel, message.author, docarray_id, idx)
        return
    elif len(message.mentions) == 1 and \
        message.mentions[0].id == client.user.id and \
        len(message.attachments) >= 1:
        for i, attachment in enumerate(message.attachments):
            sid = short_id_generator.generate()
            image_fn = f'images/{sid}.png'
            da_fn = f'image_docarrays/{sid}.bin'

            image_bytes = await attachment.read()
            try:
                image = Image.open(BytesIO(image_bytes))
            except Exception:
                message.channel.send(f'Could not load image file for attachment {i}')
                continue

            # TODO Allow non-square image sizes
            image = image.resize((512, 512)).convert('RGB')
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
            await message.channel.send(f'Attachment {attachment.filename} ({i}) uploaded by ' +
                f'{message.author.display_name} has been uploaded and given ID `{sid}`.' +
                ' To use this ID in a riff or upscale, just use 0 for the image ' +
                'index.')
            await message.channel.send(sid)


@client.event
async def on_ready():
    await client.change_presence(activity=discord.Activity(name="H4ck t3h G1bs0n"))
    print('Bot is alive')


client.run(args.token)
