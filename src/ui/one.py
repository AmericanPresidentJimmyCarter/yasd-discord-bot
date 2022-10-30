import random
import time

from typing import TYPE_CHECKING, Any

import discord

from docarray import DocumentArray

import actions

from constants import (
    BUTTON_STORE_ONE_IMAGE_BUTTONS_KEY,
    DOCARRAY_LOCATION_FN,
    OutpaintingModes,
)
from util import (
    document_to_pil,
    short_id_generator,
    write_button_store,
)

if TYPE_CHECKING:
    from client import YASDClient


class OneImageButtons(discord.ui.View):
    RIFF_ASPECT_RATIO_PLACEHOLDER_MESSAGE = 'Select Riff Outpainting'
    RIFF_STRENGTH_PLACEHOLDER_MESSAGE = 'Select Riff Strength (no effect on paint)'

    context: 'YASDClient|None' = None
    idx_parent: int|None = None
    message_id: int|None = None
    outpaint_mode: str|None = None
    pixels_height: int|None = 512
    pixels_width: int|None = 512
    prompt_input_element: 'discord.ui.TextInput|None' = None
    prompt_input_element_custom_id: str|None = None
    prompt_mask_input_element: 'discord.ui.TextInput|None' = None
    prompt_mask_input_element_custom_id: str|None = None
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
        short_id_parent: str|None=None,
        strength: float|None=None,
        timeout=None,
    ):
        super().__init__(timeout=timeout)
        self.context = context
        self.idx_parent = idx_parent # type: ignore
        if self.idx_parent is not None and type(self.idx_parent) == float:
            self.idx_parent = int(self.idx_parent)
        self.message_id = message_id
        self.outpaint_mode = outpaint_mode
        self.prompt_mask = prompt_mask
        self.short_id_parent = short_id_parent
        self.strength = strength

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

        self.pixels_width, self.pixels_height = self.original_image_sizes()

    def original_image_prompt(self):
        old_docarray_loc = DOCARRAY_LOCATION_FN(self.short_id_parent)
        da = DocumentArray.load_binary(
            old_docarray_loc, protocol='protobuf', compress='lz4'
        )

        return da[self.idx_parent].text

    def original_image_sizes(self):
        old_docarray_loc = DOCARRAY_LOCATION_FN(self.short_id_parent)
        da = DocumentArray.load_binary(
            old_docarray_loc, protocol='protobuf', compress='lz4'
        )
        loaded = document_to_pil(da[self.idx_parent])
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
        button_store_dict[BUTTON_STORE_ONE_IMAGE_BUTTONS_KEY].append(as_dict)
        write_button_store(
            self.context.button_store_path, # type: ignore
            self.context.button_store_dict, # type: ignore
        )

    @classmethod
    def from_serialized(cls,
        context: 'YASDClient',
        serialized: dict[str, Any],
    ) -> 'OneImageButtons':
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
            fib.pixels_height = int(serialized['height'])
        if serialized.get('width', None) is not None:
            fib.pixels_width = int(serialized['width'])

        fib.context = context

        return fib

    async def handle_riff(self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ):
        docarray_loc = DOCARRAY_LOCATION_FN(self.short_id_parent)
        da = DocumentArray.load_binary(
            docarray_loc, protocol='protobuf', compress='lz4'
        )

        latentless = False
        prompt = None
        prompt_mask = None
        resize = False
        sampler = None
        scale = None
        steps = None
        strength = None

        original_request = da[self.idx_parent].tags.get('request', None)
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

        if self.prompt_input_element.value: # type: ignore
            prompt = self.prompt_input_element.value # type: ignore
        if self.prompt_mask_input_element.value: # type: ignore
            prompt_mask = self.prompt_mask_input_element.value # type: ignore

        await interaction.response.defer()
        await actions.riff(
            interaction.channel,
            interaction.user,
            self.context, # type: ignore
            self.short_id_parent, # type: ignore
            self.idx_parent,
            height=self.pixels_height,
            latentless=latentless,
            outpaint_mode=self.outpaint_mode,
            prompt=prompt,
            prompt_mask=prompt_mask,
            resize=resize,
            sampler=sampler,
            scale=scale,
            seed=random.randint(1, 2 ** 32 - 1),
            steps=steps,
            strength=strength,
            width=self.pixels_width)

    @discord.ui.button(label="Riff", style=discord.ButtonStyle.blurple, row=0,
        custom_id=f'{short_id_generator()}-riff')
    async def riff_button(self, interaction: discord.Interaction,
        button: discord.ui.Button):
        await self.handle_riff(interaction, button)

    @discord.ui.button(label="Prompt Editor", style=discord.ButtonStyle.secondary,
        row=0,
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

    @discord.ui.select(placeholder=RIFF_ASPECT_RATIO_PLACEHOLDER_MESSAGE, row=1,
        custom_id=f'{short_id_generator()}-riff-select-aspect-ratio',
        options=[
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
            discord.SelectOption(label='Extend to 2:1 (landscape)', value='2:1'),
            discord.SelectOption(label='Extend to 3:2', value='3:2'),
            discord.SelectOption(label='Extend to 4:3', value='4:3'),
            discord.SelectOption(label='Extend to 1:1 (square)', value='1:1'),
            discord.SelectOption(label='Extend to 3:4', value='3:4'),
            discord.SelectOption(label='Extend to 2:3', value='2:3'),
            discord.SelectOption(label='Extend to 1:2 (portait)', value='1:2'),
            discord.SelectOption(label='Original Image Size', value='original'),
        ])
    async def select_aspect_ratio(self, interaction: discord.Interaction,
        selection: discord.ui.Select):
        (orig_width, orig_height) = self.original_image_sizes()
        self.outpaint_mode = None
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

    @discord.ui.select(placeholder=RIFF_STRENGTH_PLACEHOLDER_MESSAGE, row=2,
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
