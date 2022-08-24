import argparse
import asyncio
import json

from io import BytesIO
from typing import Union

import discord

from PIL import Image
from discord import File
from discord.ext import commands
from shortid import ShortId


parser = argparse.ArgumentParser()
parser.add_argument('token', help='Discord token')
args = parser.parse_args()

intents = discord.Intents(messages=True, message_content=True)
bot = commands.Bot(command_prefix='>', description="This is a Helper Bot", intents=intents)

currently_fetching_ai_image: Union[str, bool] = False
short_id_generator = ShortId()

JSON_IMAGE_TOOL_INPUT_FILE = 'temp_json/request.json'
JSON_IMAGE_TOOL_OUTPUT_FILE = 'temp_json/output.json'
MAX_ITERATIONS = 16
MAX_SEED = 2 ** 32 - 1
MIN_STEPS = 50
MAX_STEPS = 250


@bot.command(
    description='Create an image from a prompt\n\n' +
        'Images may be given variations with an array format by ' +
        'enclosing values in square brackets e.g. "a [red, blue] ball"\n\n' +
        'Accepts the following options in (foo=bar) format: \n' +
        'sampler: which sampling algorithm to use (k_lms or ddim, default k_lms)\n' +
        'scale: conditioning scale for prompt (1.0 to 50.0)\n' +
        'seed: conditioning scale for prompt (1 to 2^32-1)\n' +
        'seed_search: seed searching mode, enumerates 4 different seeds starting at given seed\n' +
        'steps: number of steps to perform (50 to 250)\n'
)
async def image(ctx, *, prompt):
    global currently_fetching_ai_image

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

    typ = 'prompt'
    if prompt.find('[') != -1 and prompt.find(']') != -1:
        typ = 'promptarray'
    if seed_search:
        typ = 'promptsearch'

    if currently_fetching_ai_image is not False:
        await ctx.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image}". Please be patient until I finish that.')
        return
    currently_fetching_ai_image = prompt
    await ctx.send(f'Now beginning work on "{prompt}". Please be patient until I finish that.')
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
        with open(JSON_IMAGE_TOOL_INPUT_FILE, 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE, 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']
        seeds = output.get('seeds', None)
        with open(image_loc, 'rb') as f:
            await ctx.send(
                f'Image generation for prompt "{prompt}" complete. The ID for your images is `{short_id}`.',
                file=File(f, image_loc))
            await ctx.send(short_id)
            if seeds is not None:
                seed_lst = []
                for i, _s in enumerate(seeds):
                    seed_lst.append(f'{i}: {_s}')
                seeds_str = ', '.join(seed_lst)
                await ctx.send(f'Seeds used were {seeds_str}')
    except Exception as e:
        await ctx.send(f'Got unknown error on prompt "{prompt}": {str(e)}')
    finally:
        currently_fetching_ai_image = False


@bot.command(
    description='Create an image from a generated image using its ID and an index\n\n' +
        'Accepts the following options in (foo=bar) format: \n' +
        'iterations: number of diffusion iterations (1 to 16)\n' +
        'latentless: do not compute latent embeddings from original image\n' +
        'prompt: Alter the prompt given to the image (if not specified, original is used)\n' +
        'sampler: which sampling algorithm to use (k_lms or ddim, default k_lms)\n' +
        'scale: conditioning scale for prompt (1.0 to 50.0)\n' +
        'seed: conditioning scale for prompt (1 to 2^32-1)\n' +
        'strength: strength of conditioning (0 < strength < 1)\n'
)
async def riff(ctx, docarray_id: str, idx: int, *, text=''):
    global currently_fetching_ai_image

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

    if currently_fetching_ai_image is not False:
        await ctx.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image}". Please be patient until I finish that.')
        return
    currently_fetching_ai_image = f'riffs on previous work `{docarray_id}`, index {str(idx)}'
    await ctx.send(f'Now beginning work on "riff `{docarray_id}` index {str(idx)}". Please be patient until I finish that.')
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
        with open(JSON_IMAGE_TOOL_INPUT_FILE, 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE, 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']
        with open(image_loc, 'rb') as f:
            await ctx.send(f'Image generation for riff on `{docarray_id}` index {str(idx)} complete. The ID for your new images is `{short_id}`.',
                file=File(f, image_loc))
            await ctx.send(short_id)
    except Exception as e:
        await ctx.send(f'Got unknown error on riff "{docarray_id}" index {str(idx)}: {str(e)}')
    finally:
        currently_fetching_ai_image = False


@bot.command(
    description='Create an image from an uploaded image\n\n' +
        'Accepts the following options in (foo=bar) format: \n' +
        'iterations: number of diffusion iterations (1 to 16)\n' +
        'latentless: do not compute latent embeddings from original image\n' +
        'sampler: which sampling algorithm to use (k_lms or ddim, default k_lms)\n' +
        'scale: conditioning scale for prompt (1.0 to 50.0)\n' +
        'seed: conditioning scale for prompt (1 to 2^32-1)\n' +
        'strength: strength of conditioning (0 < strength < 1)\n'
)
async def image2image(ctx, *, prompt):
    global currently_fetching_ai_image

    filename = f'images/attachment-{short_id_generator.generate()}.png'
    if len(ctx.message.attachments) != 1:
        await ctx.send('Please upload a single image with your message (square is best)')
    else:
        image_bytes = await ctx.message.attachments[0].read()
        image = Image.open(BytesIO(image_bytes))
        image = image.resize((512, 512)).convert('RGB')
        image.save(filename, format='PNG')

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

    if currently_fetching_ai_image is not False:
        await ctx.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image}". Please be patient until I finish that.')
        return

    currently_fetching_ai_image = f'image2image on prompt `{prompt}`'
    await ctx.send(f'Now beginning work on "image2image `{prompt}`". Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'filename': filename,
            'from_discord': True,
            'iterations': iterations,
            'latentless': latentless,
            'prompt': prompt,
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
            'strength': strength,
            'type': 'riff',
        }
        with open(JSON_IMAGE_TOOL_INPUT_FILE, 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE, 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']
        with open(image_loc, 'rb') as f:
            await ctx.send(f'Image generation for image2image on `{prompt}` complete. The ID for your new images is `{short_id}`.',
                file=File(f, image_loc))
            await ctx.send(short_id)
    except Exception as e:
        await ctx.send(f'Got unknown error on image2image "{prompt}": {str(e)}')
    finally:
        currently_fetching_ai_image = False


@bot.command(
    description='Upscale a generated image using its ID and index'
)
async def upscale(ctx, docarray_id: str, idx: int):
    global currently_fetching_ai_image
    if currently_fetching_ai_image is not False:
        await ctx.send(f'Sorry, I am currently working on the image prompt "{currently_fetching_ai_image}". Please be patient until I finish that.')
        return

    currently_fetching_ai_image = f'upscale on previous work `{docarray_id}`, index {str(idx)}'
    await ctx.send(f'Now beginning work on "upscale `{docarray_id}` index {str(idx)}". Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'docarray_id': docarray_id,
            'index': idx,
            'type': 'upscale',
        }
        with open(JSON_IMAGE_TOOL_INPUT_FILE, 'w') as inp:
            inp.write(json.dumps(req))
        proc = await asyncio.create_subprocess_exec(
            'python','imagetool.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        output = None
        with open(JSON_IMAGE_TOOL_OUTPUT_FILE, 'r') as out:
            output = json.load(out)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        with open(image_loc, 'rb') as f:
            await ctx.send(f'Image generation for upscale on `{docarray_id}` index {str(idx)} complete.',
                file=File(f, image_loc))
    except Exception as e:
        await ctx.send(f'Got unknown error on upscale "{docarray_id}" index {str(idx)}: {str(e)}')
    finally:
        currently_fetching_ai_image = False


@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Activity(name="H4ck t3h G1bs0n"))
    print('Bot is alive')


bot.run(args.token)
