import math
import random
import time

from typing import TYPE_CHECKING, Any

import discord

from docarray import DocumentArray

import actions

from constants import (
    BUTTON_STORE_FOUR_IMAGES_BUTTONS_KEY,
    DEFAULT_IMAGE_HEIGHT_WIDTH,
    DOCARRAY_LOCATION_FN,
    MAX_UPSCALE_SIZE,
    OutpaintingModes,
    UPSCALER_NONE,
    UPSCALER_REALESRGAN_4X,
    UPSCALER_REALESRGAN_4X_ANIME,
    UPSCALER_REALESRGAN_4X_FACE,
    UPSCALER_STABLE_1,
    UPSCALER_STABLE_2,
    UPSCALER_STABLE_3,
    UPSCALER_STABLE_4,
    UPSCALER_STABLE_5,
    UPSCALER_SWINIR,
)


from util import (
    document_to_pil,
    short_id_generator,
    write_button_store,
)

if TYPE_CHECKING:
    from client import YASDClient


class FourImageButtons(discord.ui.View):
    RIFF_ASPECT_RATIO_PLACEHOLDER_MESSAGE = 'Select Riff Aspect Ratio'
    RIFF_STRENGTH_PLACEHOLDER_MESSAGE_OLD = 'Select Riff Strength (no effect on outriff)'
    RIFF_STRENGTH_PLACEHOLDER_MESSAGE_NEW = 'Select Riff Strength (no effect on outpaint)'
    UPSCALER_PLACEHOLDER_MESSAGE = 'Select Upscaler'

    context: 'YASDClient|None' = None
    idx_parent: int|None = None
    message_id: int|None = None
    outpaint_mode: str|None = None
    pixels_height: int|None = DEFAULT_IMAGE_HEIGHT_WIDTH
    pixels_width: int|None = DEFAULT_IMAGE_HEIGHT_WIDTH
    prompt_input_element: 'discord.ui.TextInput|None' = None
    prompt_input_element_custom_id: str|None = None
    prompt_mask_input_element: 'discord.ui.TextInput|None' = None
    prompt_mask_input_element_custom_id: str|None = None
    short_id: str|None = None
    short_id_parent: str|None = None
    strength: float|None = None
    upscaler: str|None = None

    def __init__(
        self,
        *,
        context: 'YASDClient|None'=None,
        idx_parent: int|float|None=None,
        message_id: int|None=None,
        outpaint_mode: str|None=None,
        prompt: str|None=None,
        prompt_input_element_custom_id: str|None=None,
        prompt_mask: str|None=None,
        prompt_mask_input_element_custom_id: str|None=None,
        short_id: str|None=None,
        short_id_parent: str|None=None,
        strength: float|None=None,
        timeout=None,
    ):
        super().__init__(timeout=timeout)
        self.context = context
        self.idx_parent = idx_parent # type: ignore
        if self.idx_parent is not None and type(self.idx_parent) == float:
            self.idx_parent = int(self.idx_parent)
        self.outpaint_mode = outpaint_mode
        self.message_id = message_id
        self.short_id = short_id
        self.short_id_parent = short_id_parent
        self.strength = strength

        self.pixels_width, self.pixels_height = self.original_image_sizes()

        self.prompt_input_element = discord.ui.TextInput(
            custom_id=prompt_input_element_custom_id or
                f'{short_id_generator()}-riff-prompt',
            default=prompt or self.original_image_prompt(),
            label='Prompt',
            placeholder='Enter New Prompt for Riff',
            required=False,
            row=0,
        )
        self.prompt_mask_input_element = discord.ui.TextInput(
            custom_id=prompt_mask_input_element_custom_id or
                f'{short_id_generator()}-riff-prompt-mask',
            default=prompt_mask,
            label='Prompt Mask',
            placeholder='Enter Prompt For Selection (prefix with "not" for inverse)',
            required=False,
            row=1,
        )

    def original_image_prompt(self):
        old_docarray_loc = DOCARRAY_LOCATION_FN(self.short_id)
        da = DocumentArray.load_binary(
            old_docarray_loc, protocol='protobuf', compress='lz4'
        )

        return da[0].text

    def original_image_sizes(self):
        old_docarray_loc = DOCARRAY_LOCATION_FN(self.short_id)
        da = DocumentArray.load_binary(
            old_docarray_loc, protocol='protobuf', compress='lz4'
        )
        loaded = document_to_pil(da[0])
        return loaded.size

    def serialize_to_json_and_store(self, button_store_dict: dict[str, Any]):
        '''
        Store a serialized representation in the global magic json.
        '''
        as_dict = {
            'height': self.pixels_height,
            'idx_parent': self.idx_parent,
            'message_id': self.message_id,
            'outpaint_mode': self.outpaint_mode,
            'prompt': self.prompt_input_element.value or None, # type: ignore
            'prompt_input_element_custom_id':
                self.prompt_input_element.custom_id, # type: ignore
            'prompt_mask': self.prompt_mask_input_element.value or None, # type: ignore
            'prompt_mask_input_element_custom_id':
                self.prompt_mask_input_element.custom_id, # type: ignore
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
        write_button_store(
            self.context.button_store_path, # type: ignore
            self.context.button_store_dict, # type: ignore
        )

    @classmethod
    def from_serialized(cls,
        context: 'YASDClient',
        serialized: dict[str, Any],
    ) -> 'FourImageButtons':
        '''
        Return a view from a serialized representation.
        '''
        idx_parent = serialized.get('idx_parent', None)
        message_id = serialized['message_id']
        outpaint_mode = serialized.get('outpaint_mode', None)
        prompt = serialized.get('prompt', None)
        prompt_cid = serialized.get('prompt_input_element_custom_id', None)
        prompt_mask = serialized.get('prompt_mask', None)
        prompt_mask_cid = serialized.get('prompt_mask_input_element_custom_id',
            None)
        short_id = serialized['short_id']
        short_id_parent = serialized.get('short_id_parent', None)
        strength = serialized.get('strength', None)
        fib = cls(
            idx_parent=idx_parent,
            message_id=message_id,
            outpaint_mode=outpaint_mode,
            prompt=prompt,
            prompt_input_element_custom_id=prompt_cid,
            prompt_mask=prompt_mask,
            prompt_mask_input_element_custom_id=prompt_mask_cid,
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
                item_dict['label'] == fib.RIFF_STRENGTH_PLACEHOLDER_MESSAGE_OLD or \
                item_dict['label'] == fib.RIFF_STRENGTH_PLACEHOLDER_MESSAGE_NEW or \
                item_dict['label'] == fib.UPSCALER_PLACEHOLDER_MESSAGE:
                sel = mapped_to_label[item_dict['label']]
                sel.custom_id = item_dict['custom_id']
            else:
                btn = mapped_to_label[item_dict['label']]
                btn.custom_id = item_dict['custom_id']

        if serialized.get('height', None) is not None:
            fib.pixels_height = int(serialized['height'])
        if serialized.get('width', None) is not None:
            fib.pixels_width = int(serialized['width'])

        fib.context = context

        return fib

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

        prompt = None
        if self.prompt_input_element.value: # type: ignore
            prompt = self.prompt_input_element.value # type: ignore
        prompt_mask = None
        if self.prompt_mask_input_element.value: # type: ignore
            prompt_mask = self.prompt_mask_input_element.value # type: ignore

        await interaction.response.defer()
        await actions.riff(
            interaction.channel,
            interaction.user,
            self.context, # type: ignore
            self.short_id, # type: ignore
            idx,
            height=self.pixels_height,
            latentless=latentless,
            prompt=prompt,
            prompt_mask=prompt_mask,
            outpaint_mode=self.outpaint_mode,
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
            await interaction.response.send_message('No original request could be found')
            return

        width, height = self.original_image_sizes()

        if original_request['api'] == 'txt2img':
            prompt = da[0].text
            sampler = original_request['sampler']
            scale = original_request['scale']
            steps = int(original_request['steps'])
            await actions.image(
                interaction.channel,
                interaction.user,
                self.context, # type: ignore
                prompt,
                height=height,
                sampler=sampler,
                scale=scale,
                steps=steps,
                width=width)
        if original_request['api'] == 'stablediffuse':
            latentless = original_request['latentless']
            outpaint_mode = original_request.get('outpaint_mode', None)
            prompt = da[0].text
            prompt_mask = original_request.get('prompt_mask', None)
            resize = original_request.get('resize', False)
            sampler = original_request['sampler']
            scale = original_request['scale']
            steps = int(original_request['steps'])
            strength = original_request['strength']

            await actions.riff(
                interaction.channel,
                interaction.user,
                self.context, # type: ignore
                self.short_id_parent, # type: ignore
                self.idx_parent, # type: ignore
                height=height,
                latentless=latentless,
                prompt_mask=prompt_mask,
                outpaint_mode=outpaint_mode,
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

        prompt = None
        if self.prompt_input_element.value: # type: ignore
            prompt = self.prompt_input_element.value # type: ignore

        await actions.upscale(
            interaction.channel,
            interaction.user,
            self.context, # type: ignore
            self.short_id, # type: ignore
            idx,
            prompt=prompt,
            upscaler=self.upscaler)

    @discord.ui.button(label="Riff 0", style=discord.ButtonStyle.blurple, row=0,
        custom_id=f'{short_id_generator()}-riff-0')
    async def riff_button_0(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_riff(interaction, button, 0)

    @discord.ui.button(label="Riff 1", style=discord.ButtonStyle.blurple, row=0,
        custom_id=f'{short_id_generator()}-riff-1')
    async def riff_button_1(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_riff(interaction, button, 1)

    @discord.ui.button(label="Upscale 0", style=discord.ButtonStyle.green, row=0,
        custom_id=f'{short_id_generator()}-upscale-0')
    async def upscale_button_0(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_upscale(interaction, button, 0)

    @discord.ui.button(label="Upscale 1", style=discord.ButtonStyle.green, row=0,
        custom_id=f'{short_id_generator()}-upscale-1')
    async def upscale_button_1(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_upscale(interaction, button, 1)

    @discord.ui.button(label="Retry", style=discord.ButtonStyle.secondary, row=0,
        custom_id=f'{short_id_generator()}-riff-3')
    async def retry_button(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_retry(interaction, button)

    @discord.ui.button(label="Riff 2", style=discord.ButtonStyle.blurple, row=1,
        custom_id=f'{short_id_generator()}-riff-2')
    async def riff_button_2(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_riff(interaction, button, 2)

    @discord.ui.button(label="Riff 3", style=discord.ButtonStyle.blurple, row=1,
        custom_id=f'{short_id_generator()}-riff-3')
    async def riff_button_3(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_riff(interaction, button, 3)

    @discord.ui.button(label="Upscale 2", style=discord.ButtonStyle.green, row=1,
        custom_id=f'{short_id_generator()}-upscale-2')
    async def upscale_button_2(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_upscale(interaction, button, 2)

    @discord.ui.button(label="Upscale 3", style=discord.ButtonStyle.green, row=1,
        custom_id=f'{short_id_generator()}-upscale-3')
    async def upscale_button_3(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_upscale(interaction, button, 3)

    @discord.ui.button(label="Prompt Editor", style=discord.ButtonStyle.secondary,
        row=1,
        custom_id=f'{short_id_generator()}-prompt-modal')
    async def prompt_editor_button(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        # discordpy docs insists this needs to be a subclass, but that is
        # a lie.
        async def on_submit(modal_interaction: discord.Interaction):
            await modal_interaction.response.defer()
        modal = discord.ui.Modal(title='Prompt Editor',
            custom_id=f'prompt-modal-{short_id_generator()}')
        setattr(modal, 'on_submit', on_submit)
        modal.add_item(self.prompt_input_element) # type: ignore
        modal.add_item(self.prompt_mask_input_element) # type: ignore
        await interaction.response.send_modal(modal)

        # wait for user to submit the modal
        timed_out = await modal.wait()
        if timed_out:
            return
        self.prompt_input_element.default = self.prompt_input_element.value # type: ignore
        self.prompt_mask_input_element.default = self.prompt_mask_input_element.value # type: ignore

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
            discord.SelectOption(label='Extend 25% on all sides',
                value=OutpaintingModes.OUTPAINT_25_ALL.value),
            discord.SelectOption(label='Extend 25% left',
                value=OutpaintingModes.OUTPAINT_25_LEFT.value),
            discord.SelectOption(label='Extend 25% right',
                value=OutpaintingModes.OUTPAINT_25_RIGHT.value),
            discord.SelectOption(label='Extend 25% up',
                value=OutpaintingModes.OUTPAINT_25_UP.value),
            discord.SelectOption(label='Extend 25% down',
                value=OutpaintingModes.OUTPAINT_25_DOWN.value),
            discord.SelectOption(label='Original Image Size', value='original'),
        ])
    async def select_aspect_ratio(self, interaction: discord.Interaction,
        selection: discord.ui.Select):
        (orig_width, orig_height) = self.original_image_sizes()
        self.outpaint_mode = None
        selected = selection.values
        sel = selected[0]
        max_size = MAX_UPSCALE_SIZE
        if sel == '2:1':
            self.pixels_height = int(max_size / 2)
            self.pixels_width = max_size
        if sel == '3:2': # ish
            self.pixels_height = 16 * math.ceil(max_size * (2/3) * (1/16))
            self.pixels_width = max_size
        if sel == '4:3':
            self.pixels_height = 16 * math.ceil(max_size * (3/4) * (1/16))
            self.pixels_width = max_size
        if sel == '1:1':
            self.pixels_height = max_size
            self.pixels_width = max_size
        if sel == '3:4':
            self.pixels_height = max_size
            self.pixels_width = 16 * math.ceil(max_size * (3/4) * (1/16))
        if sel == '2:3': # ish
            self.pixels_height = max_size
            self.pixels_width = 16 * math.ceil(max_size * (2/3) * (1/16))
        if sel == '1:2':
            self.pixels_height = max_size
            self.pixels_width = int(max_size / 2)
        if sel in [
            OutpaintingModes.OUTPAINT_25_ALL,
            OutpaintingModes.OUTPAINT_25_LEFT,
            OutpaintingModes.OUTPAINT_25_RIGHT,
            OutpaintingModes.OUTPAINT_25_UP,
            OutpaintingModes.OUTPAINT_25_DOWN,
        ]:
            self.pixels_width = orig_width
            self.pixels_height = orig_height
            self.outpaint_mode = sel
        if sel == 'original':
            self.pixels_width = orig_width
            self.pixels_height = orig_height
        await interaction.response.defer()

    @discord.ui.select(placeholder=RIFF_STRENGTH_PLACEHOLDER_MESSAGE_NEW, row=3,
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

    @discord.ui.select(placeholder=UPSCALER_PLACEHOLDER_MESSAGE, row=4,
        custom_id=f'{short_id_generator()}-upscaler-select',
        options=[
            discord.SelectOption(label='SwinIR (default, photos and art)',
                value=UPSCALER_SWINIR),
            discord.SelectOption(label='RealESRGAN (photos and art)',
                value=UPSCALER_REALESRGAN_4X),
            discord.SelectOption(label='RealESRGAN Face-Fixing (photos)',
                value=UPSCALER_REALESRGAN_4X_FACE),
            discord.SelectOption(label='RealESRGAN Anime (line art and anime)',
                value=UPSCALER_REALESRGAN_4X_ANIME),
            discord.SelectOption(label='Diffusion Upscale (0.1 strength)',
                value=UPSCALER_STABLE_1),
            discord.SelectOption(label='Diffusion Upscale (0.2 strength)',
                value=UPSCALER_STABLE_2),
            discord.SelectOption(label='Diffusion Upscale (0.3 strength)',
                value=UPSCALER_STABLE_3),
            discord.SelectOption(label='Diffusion Upscale (0.4 strength)',
                value=UPSCALER_STABLE_4),
            discord.SelectOption(label='Diffusion Upscale (0.5 strength)',
                value=UPSCALER_STABLE_5),
            discord.SelectOption(label='No Upscale (gives options to edit image)',
                value=UPSCALER_NONE),
        ])
    async def select_upscaler(self, interaction: discord.Interaction,
        selection: discord.ui.Select):
        selected = selection.values
        if selected[0] is None or selected[0] == 'swinir':
            self.upscaler = None
        else:
            self.upscaler = selected[0]

        await interaction.response.defer()
