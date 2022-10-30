import pathlib

from argparse import Namespace
from typing import Any, Callable

import discord

from discord import app_commands


class YASDClient(discord.Client):
    '''
    The root client for YASD discord bot.
    '''
    button_store_dict: dict[str, Any]|None = None
    button_store_path: pathlib.Path|None = None
    cli_args: Namespace|None = None
    currently_fetching_ai_image: dict[str, bool|str|list[str]]|None = None
    guild_id: int|None = None
    prompt_check_fn: Callable|None = lambda x: x
    safety_checker: Callable|None = None
    safety_feature_extractor: Callable|None = None
    user_image_generation_nonces: dict[str, int]|None = None

    def __init__(
        self,
        intents,
        button_store_dict=None,
        button_store_path: pathlib.Path=None,
        cli_args: Namespace=None,
        currently_fetching_ai_image: dict[str, bool|str|list[str]]=None,
        guild_id: int|None=None,
        prompt_check_fn: Callable=None,
        safety_checker: Callable=None,
        safety_feature_extractor: Callable=None,
        user_image_generation_nonces: dict[str, int]=None,
    ):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

        self.button_store_dict = button_store_dict
        self.button_store_path = button_store_path
        self.cli_args = cli_args
        self.currently_fetching_ai_image = currently_fetching_ai_image
        self.guild_id = guild_id
        self.prompt_check_fn = prompt_check_fn
        self.safety_checker = safety_checker
        self.safety_feature_extractor = safety_feature_extractor
        self.user_image_generation_nonces = user_image_generation_nonces

    async def setup_hook(self):
        guild_id = None
        if self.guild_id is not None:
            guild_id =  discord.Object(id=self.guild_id)
        if guild_id is not None:
            self.tree.copy_global_to(guild=guild_id)
            await self.tree.sync(guild=guild_id)
