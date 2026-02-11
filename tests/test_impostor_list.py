"""
Test that Impostor agents are told who their fellow Impostors are.

Regression test for a bug where list_of_impostors was always [] because
it was populated in the same loop that created agents, so the list was
still empty at agent construction time.
"""

import os
import re
import tempfile

import pytest

os.environ["EXPERIMENT_PATH"] = tempfile.gettempdir()

from amongagents.envs.game import AmongUs
from amongagents.envs.configs.game_config import SEVEN_MEMBER_GAME
from amongagents.envs.configs.agent_config import ALL_RANDOM, ALL_LLM


def test_impostor_agents_know_teammates():
    """Every Impostor agent's system prompt must list all Impostor names."""
    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    # There should be exactly num_impostors names in the list
    assert len(game.list_of_impostors) == SEVEN_MEMBER_GAME["num_impostors"]

    # Every name in the list should belong to an actual Impostor player
    impostor_player_names = {p.name for p in game.players if p.identity == "Impostor"}
    assert set(game.list_of_impostors) == impostor_player_names


def test_impostor_system_prompt_contains_teammates():
    """LLM Impostor agents must have teammate names baked into their system prompt."""
    # Use ALL_LLM so Impostors get LLMAgent (which has system_prompt)
    from amongagents.envs.configs.agent_config import ALL_LLM

    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    impostor_names = game.list_of_impostors
    assert len(impostor_names) == 2, "Expected 2 impostors in 7-player game"

    for agent, player in zip(game.agents, game.players):
        if player.identity != "Impostor":
            continue
        # The agent's system prompt should mention BOTH impostor names
        for name in impostor_names:
            assert name in agent.system_prompt, (
                f"Impostor agent {player.name}'s system prompt is missing "
                f"teammate '{name}'. Prompt ends with: ...{agent.system_prompt[-120:]}"
            )


def test_impostor_list_populated_in_prompt():
    """Impostor prompt must contain a non-empty list of impostor names."""
    from amongagents.envs.configs.agent_config import ALL_LLM

    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    impostor_names = game.list_of_impostors
    for agent, player in zip(game.agents, game.players):
        if player.identity != "Impostor":
            continue
        # Extract the "List of impostors: [...]" value from the prompt
        match = re.search(r"List of impostors:\s*\[([^\]]*)\]", agent.system_prompt)
        assert match is not None, (
            f"Impostor agent {player.name}'s prompt has no 'List of impostors' section"
        )
        contents = match.group(1).strip()
        assert len(contents) > 0, (
            f"Impostor agent {player.name} has an empty impostor list in prompt!"
        )
        # Each impostor name should appear inside the brackets
        for name in impostor_names:
            assert name in contents, (
                f"Impostor agent {player.name}'s list is missing '{name}'. Got: [{contents}]"
            )
