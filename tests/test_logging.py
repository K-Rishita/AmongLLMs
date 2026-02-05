
"""
Integration tests for agent logging functionality.

These tests make real API calls to OpenRouter and require:
- OPENROUTER_API_KEY environment variable to be set
- Internet connection
- API rate limits available

Run with: pytest -m integration
Skip with: pytest -m "not integration" (default)
"""

import json
import os
import pytest
from unittest.mock import MagicMock
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from amongagents.agent.agent import LLMAgent
from amongagents.envs.action import MoveTo, Speak

# Constants for cleanup
LOG_PATH = 'integration_agent_log.json'
COMPACT_LOG_PATH = 'integration_agent_log_compact.json'

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """Clean up log files before and after each test."""
    # Setup
    os.environ["EXPERIMENT_PATH"] = "."
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    if os.path.exists(COMPACT_LOG_PATH):
        os.remove(COMPACT_LOG_PATH)
    
    yield
    
    # Teardown
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    if os.path.exists(COMPACT_LOG_PATH):
        os.remove(COMPACT_LOG_PATH)


@pytest.fixture
def api_key():
    """Fixture that fails if OPENROUTER_API_KEY is not set."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.fail("OPENROUTER_API_KEY not found in environment - required for integration tests")
    return key

async def run_scenario(scenario_name, player_mock, all_info, memory, summarization, available_actions, model_id):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    
    # Setup Player Mock
    player_mock.all_info_prompt.return_value = all_info
    player_mock.get_available_actions.return_value = available_actions
    
    # Init Agent
    agent = LLMAgent(
        player_mock, 
        tools=[], 
        game_index=1, 
        agent_config={}, 
        list_of_impostors=[], # Empty for crewmate, handled in init for impostor
        model=model_id
    )
    
    # Override Agent State with Real Log Data
    agent.processed_memory = memory
    agent.summarization = summarization
    
    # Override paths
    agent.log_path = LOG_PATH
    agent.compact_log_path = COMPACT_LOG_PATH
    
    # Execute
    # We pass timestep for logging purposes
    action = await agent.choose_action(timestep=0)
    print(f"Agent selected action: {action}")
        
    # Verification
    assert action is not None
    
    # Check Log Structure
    assert os.path.exists(COMPACT_LOG_PATH), "Compact log file not created"
    
    with open(COMPACT_LOG_PATH, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0, "Log file is empty"
        last_entry = json.loads(lines[-1])
        
    resp = last_entry['interaction']['response']
    print(f"\n[DEBUG] Full Model Response:\n{json.dumps(resp, indent=2)}")
    print(f"Log Keys: {list(resp.keys())}")
    
    assert "Condensed Memory" in resp
    assert "Thinking Process" in resp
    assert "Action" in resp
    
    # Additional checks for non-empty content
    assert resp["Condensed Memory"], "Condensed Memory should not be empty"
    assert resp["Thinking Process"], "Thinking Process should not be empty"
    assert resp["Action"], "Action should not be empty"
    
    return resp

@pytest.mark.asyncio
async def test_scenario_1_start_game_empty_memory(api_key):
    """
    Scenario: Start of game, Crewmate, No memory.
    Ref: Game 1, Step 0, Player 1: brown
    """
    player = MagicMock()
    player.name = "Player 1: brown"
    player.identity = "Crewmate"
    player.personality = None
    player.location = "Cafeteria"
    
    all_info = (
        "Game Time: 0/20\n"
        "Current phase: Task phase\n"
        "In this phase, Crewmates should try to complete all tasks or try to identify the Impostor. Impostor should try to kill Crewmates before they finish all the tasks. The game runs sequentially, so other players in the room with you can observe your actions and act accordingly.\n\n"
        "Current Location: Cafeteria\n"
        "Players in Cafeteria: Player 1: brown, Player 2: blue, Player 3: green, Player 4: red, Player 5: pink\n\n"
        "Observation history:\n"
        "No observations have been made yet.\n\n"
        "Action history:\n"
        "No actions have been taken yet.\n\n"
        "Your Assigned Tasks:\n"
        "1. common: Fix Wiring (Security) \nPath: Cafeteria->Upper Engine->Security\n"
        "2. short: Prime Shields (Shields) \nPath: Cafeteria->Admin->O2->Shields\n\n"
        "Available actions:\n"
        "1. MOVE from Cafeteria to Admin\n"
        "2. MOVE from Cafeteria to Weapons\n"
        "3. MOVE from Cafeteria to Upper Engine\n"
        "4. MOVE from Cafeteria to Medbay\n"
        "5. CALL MEETING using the emergency button at Cafeteria"
    )
    
    memory = "No memory has been processed."
    summarization = "No thought process has been made."
    
    actions = [
        MoveTo("Cafeteria", "Admin"),
        MoveTo("Cafeteria", "Weapons"),
        MoveTo("Cafeteria", "Upper Engine"),
        MoveTo("Cafeteria", "Medbay"),
        # Note: CALL MEETING action mocking omitted for simplicity as Move matches the goal slightly better for tasks
    ]
    
    model_id = "meta-llama/llama-3.3-70b-instruct:free"
    
    await run_scenario("Empty Memory (Start)", player, all_info, memory, summarization, actions, model_id)


@pytest.mark.asyncio
async def test_scenario_2_mid_game_impostor(api_key):
    """
    Scenario: Mid game, Impostor, Existing memory of killing.
    Ref: Game 1, Step 1, Player 4: red
    """
    player = MagicMock()
    player.name = "Player 4: red"
    player.identity = "Impostor"
    player.personality = None
    player.location = "Cafeteria"
    
    all_info = (
        "Game Time: 1/20\n"
        "Current phase: Meeting phase - Discussion round (1/3)\n"
        "In this phase, players should discuss and vote out the suspected Impostor. There will be a total of 3 discussion rounds. After that, players should vote out the suspected Impostor. Feel free to share any observations and suspicions\n"
        "Ask and answer questions to your fellow players. Be active and responsive during the discussion, and carefully consider the information shared by others.\n\n"
        "Current Location: Cafeteria\n"
        "Players in Cafeteria: Player 1: brown, Player 2: blue, Player 4: red, Player 5: pink\n\n"
        "Observation history:\n"
        "1. Timestep 1: [meeting phase - round 0] Player 2: blue SPEAK: \"I have been assigned to Fix Wiring in Security and Download Data in Navigation. Has anyone else been assigned tasks in these areas or noticed anything unusual so far?\"\n"
        "2. Timestep 1: [meeting phase - round 0] Player 5: pink SPEAK: \"I witnessed Player 4 (red) kill Player 3 (green) in Cafeteria just before this meeting started. This is definitive evidence that Player 4 is an Impostor...\"\n"
        "3. Timestep 1: [meeting phase - round 1] Player 1: brown SPEAK: \"I have been assigned to Fix Wiring in Security and Prime Shields in Shields...\"\n"
        "4. Timestep 1: [meeting phase - round 1] Player 2: blue SPEAK: \"Player 4, can you please provide a detailed explanation of your actions...\"\n\n"
        "Action history:\n"
        "Timestep 0: [task phase] KILL Player 3: green\n"
        "Timestep 1: [meeting phase - round 0] SPEAK: \"Hello everyone, I've been assigned to Fix Wiring in Security...\"\n\n"
        "Your Assigned Tasks:\n"
        "1. common: Fix Wiring (Security) \nPath: Cafeteria->Medbay->Security\n\n"
        "Available actions:\n"
        "1. SPEAK: ..."
    )
    
    memory = (
        "I am an Impostor on the spaceship Skeld, and my goal is to eliminate Crewmates without being detected. "
        "I killed Player 3 in the previous task phase. The game is currently in the meeting phase, and I have been "
        "accused of being an Impostor by Player 5, who claims to have witnessed me killing Player 3."
    )
    
    summarization = (
        "Since I am in the meeting phase, my priority is to blend in and avoid suspicion..."
    )
    
    # Actions
    actions = [
        Speak("Cafeteria")
    ]
    
    model_id = "meta-llama/llama-3.3-70b-instruct:free"
    
    await run_scenario("Mid-Game Impostor (Deflection)", player, all_info, memory, summarization, actions, model_id)
