from typing import TYPE_CHECKING, Optional

import discord

from constants import (
    DISCORD_EMBED_MAX_LENGTH,
    REGEX_FOR_ID,
    UPSCALER_NONE,
    UPSCALER_STABLE_1,
    UPSCALER_STABLE_2,
    UPSCALER_STABLE_3,
    UPSCALER_STABLE_4,
    UPSCALER_STABLE_5,
)
from serializers import (
    serialize_image_request,
    serialize_interpolate_request,
    serialize_riff_request,
    serialize_upscale_request,
)
from ui import (
    FourImageButtons,
    OneImageButtons,
)
from util import (
    bump_nonce_and_return,
    check_queue_and_maybe_write_to,
    check_restricted_to_channel,
    check_user_joined_at,
    complete_request,
    seed_from_docarray_id,
    spawn_image_tool_instance,
    to_discord_file_and_maybe_check_safety,
    tweak_docarray_tags_request,
)


if TYPE_CHECKING:
    from ..client import YASDClient


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
        embed.add_field(name='Command Executed', value=serialized_cmd, inline=False)
    else:
        for idx, chunk in enumerate(serialized_chunks):
            embed.add_field(
                name='Command Executed' if not idx else 'Continued',
                value=chunk, inline=False)

    embed.set_thumbnail(url=work_msg.attachments[0].url)
    await channel.send(f'Job completed for <@{author_id}>.', embed=embed)



async def image(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,
    context: 'YASDClient',

    prompt: str,

    height: Optional[int]=None,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    seed_search: bool=None,
    steps: Optional[int]=None,
    width: Optional[int]=None,
):
    author_id = str(user.id)

    if not await check_restricted_to_channel(context, channel):
        return

    if steps is None:
        steps = context.cli_args.default_steps # type: ignore

    if sampler is None and context.cli_args.default_sampler: # type: ignore
        sampler = context.cli_args.default_sampler # type: ignore

    short_id = None
    typ = 'prompt'
    if prompt.find('[') != -1 and prompt.find(']') != -1:
        typ = 'promptarray'
    if seed_search:
        typ = 'promptsearch'

    prompt = await context.prompt_check_fn(prompt, author_id, channel) # type: ignore
    if prompt is False:
        return

    if not await check_user_joined_at(context.cli_args.hours_needed, channel, # type: ignore
        user):
        return

    queue_message = prompt
    if not await check_queue_and_maybe_write_to(context, channel, author_id,
        queue_message):
        return

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
        nonce = bump_nonce_and_return(context.user_image_generation_nonces, # type: ignore
            author_id)
        output = await spawn_image_tool_instance(author_id, nonce, req)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']
        tweak_docarray_tags_request(short_id, {
            'user_id': user.id,
        })
        seeds = output.get('seeds', None)

        file = to_discord_file_and_maybe_check_safety(
            context.cli_args.auto_spoiler, # type: ignore
            image_loc,
            context.safety_feature_extractor,
            context.safety_checker)
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
            btns = FourImageButtons(context=context, message_id=work_msg.id,
                short_id=short_id)
            btns.serialize_to_json_and_store(context.button_store_dict) # type: ignore
            context.add_view(btns, message_id=work_msg.id)
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
        import traceback
        traceback.print_exc()
        await channel.send(f'Got unknown error on prompt "{prompt}": {str(e)}')
    finally:
        complete_request(context, author_id, queue_message)

    return short_id


async def riff(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,
    context: 'YASDClient',

    docarray_id: str,
    idx: int,

    height: Optional[int]=None,
    iterations: Optional[int]=None,
    latentless: bool=False,
    outpaint_mode: Optional[str]=None,
    prompt: Optional[str]=None,
    prompt_mask: Optional[str]=None,
    resize: bool=False,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    steps: Optional[int]=None,
    strength: Optional[float]=None,
    width: Optional[int]=None,
):
    author_id = str(user.id)

    if steps is None:
        steps = context.cli_args.default_steps # type: ignore

    if sampler is None and context.cli_args.default_sampler: # type: ignore
        sampler = context.cli_args.default_sampler # type: ignore

    if not await check_restricted_to_channel(context, channel):
        return

    if REGEX_FOR_ID.match(docarray_id) is None:
        await channel.send(f'Got invalid docarray ID \'{docarray_id}\'')
        return

    short_id = None
    queue_message = f'riffs on previous work `{docarray_id}`, index {str(idx)}'
    if not await check_queue_and_maybe_write_to(context, channel, author_id,
        queue_message):
        return

    prompt = await context.prompt_check_fn(prompt, author_id, channel) # type: ignore
    if prompt is False:
        return

    if not await check_user_joined_at(context.cli_args.hours_needed, channel, # type: ignore
        user):
        return

    work_msg = await channel.send(f'Now beginning work on "riff `{docarray_id}` index {str(idx)}" for <@{author_id}>. Please be patient until I finish that.')
    try:
        # Make the request in the filesystem pipeline
        req = {
            'docarray_id': docarray_id,
            'height': height,
            'index': idx,
            'iterations': iterations,
            'latentless': bool(latentless),
            'max_image_size': context.cli_args.max_image_size, # type: ignore
            'outpaint_mode': outpaint_mode,
            'prompt': prompt,
            'prompt_mask': prompt_mask,
            'resize': bool(resize),
            'sampler': sampler,
            'scale': scale,
            'seed': seed,
            'steps': steps,
            'strength': strength,
            'type': 'riff',
            'width': width,
        }
        nonce = bump_nonce_and_return(context.user_image_generation_nonces, # type: ignore
            author_id)
        output = await spawn_image_tool_instance(author_id, nonce, req)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']
        tweak_docarray_tags_request(short_id, {
            'original_image': docarray_id,
            'user_id': user.id,
        })

        file = to_discord_file_and_maybe_check_safety(
            context.cli_args.auto_spoiler, # type: ignore
            image_loc,
            context.safety_feature_extractor,
            context.safety_checker)
        btns = FourImageButtons(context=context, message_id=work_msg.id,
            idx_parent=idx, short_id=short_id, short_id_parent=docarray_id)
        btns.serialize_to_json_and_store(context.button_store_dict) # type: ignore
        context.add_view(btns, message_id=work_msg.id)
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
            outpaint_mode=outpaint_mode,
            prompt=prompt,
            prompt_mask=prompt_mask,
            resize=bool(resize),
            sampler=sampler,
            scale=scale,
            seed=seed_from_docarray_id(short_id),
            steps=steps,
            strength=strength,
            width=width)
        await send_alert_embed(channel, author_id, work_msg, serialized_cmd)
    except Exception as e:
        import traceback
        traceback.print_exc()
        await channel.send(f'Got unknown error on riff "{docarray_id}" index {str(idx)}: {str(e)}')
    finally:
        complete_request(context, author_id, queue_message)

    return short_id

async def interpolate(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,
    context: 'YASDClient',

    prompt1: str,
    prompt2: str,

    height: Optional[int]=None,
    resample_prior: bool=True,
    sampler: Optional[str]=None,
    scale: Optional[float]=None,
    seed: Optional[int]=None,
    steps: Optional[int]=None,
    strength: Optional[float]=None,
    width: Optional[int]=None,
):
    author_id = str(user.id)
    
    prompt1 = prompt1.strip()
    prompt2 = prompt2.strip()

    if steps is None:
        steps = context.cli_args.default_steps # type: ignore

    if sampler is None and context.cli_args.default_sampler: # type: ignore
        sampler = context.cli_args.default_sampler # type: ignore

    if not await check_restricted_to_channel(context, channel):
        return

    queue_message = f'interpolate on prompt {prompt1} to {prompt2}'
    if not await check_queue_and_maybe_write_to(context, channel, author_id,
        queue_message):
        return

    prompt1 = await context.prompt_check_fn(prompt1, author_id, channel) # type: ignore
    if prompt1 is False:
        return

    prompt2 = await context.prompt_check_fn(prompt2, author_id, channel) # type: ignore
    if prompt2 is False:
        return

    if not await check_user_joined_at(context.cli_args.hours_needed, channel, # type: ignore
        user):
        return

    short_id = None
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
        nonce = bump_nonce_and_return(context.user_image_generation_nonces, # type: ignore
            author_id)
        output = await spawn_image_tool_instance(author_id, nonce, req)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']
        short_id = output['id']
        tweak_docarray_tags_request(short_id, {
            'user_id': user.id,
        })

        file = to_discord_file_and_maybe_check_safety(
            context.cli_args.auto_spoiler, # type: ignore
            image_loc,
            context.safety_feature_extractor,
            context.safety_checker)
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
        import traceback
        traceback.print_exc()
        await channel.send(f'Got unknown error on interpolate `{prompt1}` to `{prompt2}`: {str(e)}')
    finally:
        complete_request(context, author_id, queue_message)

    return short_id


async def upscale(
    channel: discord.abc.GuildChannel,
    user: discord.abc.User,
    context: 'YASDClient',

    docarray_id: str,
    idx: int,

    prompt: Optional[str]=None, # For diffusion upscalers
    upscaler: Optional[str]=None,
):
    author_id = str(user.id)

    if not await check_restricted_to_channel(context, channel):
        return

    if REGEX_FOR_ID.match(docarray_id) is None:
        await channel.send(f'Got invalid docarray ID \'{docarray_id}\'')
        return

    queue_message = f'upscale on previous work `{docarray_id}`, index {str(idx)}'
    if not await check_queue_and_maybe_write_to(context, channel, author_id,
        queue_message):
        return

    if not await check_user_joined_at(context.cli_args.hours_needed, channel, # type: ignore
        user):
        return

    work_msg = await channel.send(f'Now beginning work on "upscale `{docarray_id}` index {str(idx)}" for <@{author_id}>. Please be patient until I finish that.')
    completed = False
    try:
        # Make the request in the filesystem pipeline
        req = {
            'docarray_id': docarray_id,
            'index': idx,
            'max_image_size': context.cli_args.max_image_size, # type: ignore
            'prompt': prompt,
            'type': 'upscale',
            'upscaler': upscaler,
        }
        nonce = bump_nonce_and_return(context.user_image_generation_nonces, # type: ignore
            author_id)
        output = await spawn_image_tool_instance(author_id, nonce, req)

        err = output.get('error', None)
        if err is not None:
            raise Exception(err)
        image_loc = output['image_loc']

        file = to_discord_file_and_maybe_check_safety(
            context.cli_args.auto_spoiler, # type: ignore
            image_loc,
            context.safety_feature_extractor,
            context.safety_checker)

        view = None
        stable_upscalers = [UPSCALER_STABLE_1, UPSCALER_STABLE_2,
            UPSCALER_STABLE_3, UPSCALER_STABLE_4, UPSCALER_STABLE_5]
        if upscaler in [UPSCALER_NONE, *stable_upscalers]:
            short_id_parent = docarray_id
            idx_parent = idx
            if upscaler in stable_upscalers:
                short_id_parent = output['id']
                idx_parent = 0
            view = OneImageButtons(context=context, message_id=work_msg.id,
                short_id_parent=short_id_parent, idx_parent=idx_parent)
            view.serialize_to_json_and_store(context.button_store_dict) # type: ignore
            context.add_view(view, message_id=work_msg.id)

        work_msg = await work_msg.edit(
            content=f'Image generation for upscale on `{docarray_id}` index {str(idx)} for <@{author_id}> complete.',
            attachments=[file],
            view=view)

        serialized_cmd = serialize_upscale_request(docarray_id=docarray_id,
            idx=idx, upscaler=upscaler)
        await send_alert_embed(channel, author_id, work_msg, serialized_cmd)
        completed = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        await channel.send(f'Got unknown error on upscale "{docarray_id}" index {str(idx)}: {str(e)}')
    finally:
        complete_request(context, author_id, queue_message)

    return completed
