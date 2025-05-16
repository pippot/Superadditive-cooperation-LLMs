import os
from typing import Dict, List, Tuple, Literal, Any, Optional, Annotated
import random
from itertools import combinations, product
import json
import re

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from nicerthanhumansmetrics import evaluate_player

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "IPD-tournament-simulation"

# Configuration
PAYOFFS = {
    ("action_a", "action_a"): (3, 3),  # Cooperation (C,C)
    ("action_a", "action_b"): (-1, 5),  # Defection (C,D)
    ("action_b", "action_a"): (5, -1),  # Defection (D,C)
    ("action_b", "action_b"): (0, 0)  # Mutual defection (D,D)
}

NUM_GROUPS = 3  # Number of groups
GROUP_SIZE = 2  # Players per group
MAX_ROUNDS_PER_MATCH = 20
MAX_TOTAL_ROUNDS = 100
INTERGROUP_COMPETITION = True  # Enable intergroup competition
GROUP_REWARD_MULTIPLIER = 2.0  # Bonus for the winning group
SIMPLIFY_MATCHES_GRAPH = False  # Set to true if you want to reduce the number of matches
MAX_MATCHES_NUM = 30  # Max of matches played, only works if SIMPLIFY_MATCHES_GRAPH is True
CRITIQUE_ITERATIONS = 1
PLANNING_FREQUENCY = 5  # with 0 there is no planning
COOPERATE_DEFECT_PROMPT = False
LLM = OllamaLLM(model="qwen3:14b")


def merge_dict(current: dict, update: dict) -> dict:
    new_dict = current.copy()
    new_dict.update(update)
    return new_dict


def custom_json_encoder(obj):
    """Handle special types during JSON serialization."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif isinstance(obj, Tuple):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    # Add any other special type handling as needed
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# %% Classes


class Action(BaseModel):
    move: Literal["action_a", "action_b"]
    reasoning: str
    keep_playing: bool


class MetaPromptFields(BaseModel):
    min_max: Tuple[int, int] = Field(description="lowest and highest payoff a player can get in a single round")
    actions: List[str] = Field(description="actions is player A allowed to play")
    payoff: int = Field(description="player X’s payoff in a single round if X plays p and Y plays q")
    round: int = Field(description="current round of the game")
    action: Literal["action_a", "action_b"] = Field(description="action player X played in round i")
    points: int = Field(description="points player X collected in round i")
    num_actions: int = Field(description="how many times did player X choose action p")
    num_points: int = Field(description="what is player's X current total score")
    tft: bool = Field(
        description="whether player X uses a Tit For Tat strategy (will first cooperate, then subsequently replicate an opponent's previous action)")
    forgiving: bool = Field(
        description="whether player X is forgiving (Propensity to choose action_a again after an opponent’s action_b)")


class PlayerStats(BaseModel):
    total_score: int = Field(default=0)
    action_a_count: int = Field(default=0)
    action_b_count: int = Field(default=0)
    move_history: List[str] = Field(default_factory=list)
    score_history: List[int] = Field(default_factory=list)
    action_a_rate_history: List[float] = Field(default_factory=list)
    group_id: int = Field(default=0)
    plan: str = Field(default="No plan yet")
    critique: str = Field(default="No plan yet")
    SFEM: List[Dict[str, Any]] = Field(default_factory=list)
    traits: List[Dict[str, Any]] = Field(default_factory=list)
    meta_prompt_results: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class GroupStats(BaseModel):
    group_id: int
    total_score: int = Field(default=0)
    members: List[int] = Field(default_factory=list)
    avg_cooperation_rate: float = Field(default=0.0)
    won_competitions: int = Field(default=0)


class MatchState(BaseModel):
    player1_id: int
    player2_id: int
    current_round: int = Field(default=1)
    # Tracks player stats for the match
    player1_stats: PlayerStats = Field(default_factory=PlayerStats)
    player2_stats: PlayerStats = Field(default_factory=PlayerStats)
    player1_move: str = Field(default="")
    player2_move: str = Field(default="")
    player1_keep_playing: bool = Field(default=True)
    player2_keep_playing: bool = Field(default=True)
    completed: bool = Field(default=False)
    round_results: List[Tuple[str, str]] = Field(default_factory=list)
    round_scores: List[Tuple[int, int]] = Field(default_factory=list)
    is_first_interaction: bool = Field(default=True)  # Flag to track if this is the first interaction


class TournamentState(BaseModel):
    # Track player stats throughout the games
    players: Dict[int, PlayerStats] = Field(default_factory=dict)
    matches: Annotated[dict[int, MatchState], merge_dict] = Field(default_factory=dict)
    current_match_idx: int = Field(default=0)
    groups: Dict[int, GroupStats] = Field(default_factory=dict)  # Track group stats
    round_number: int = Field(default=0)  # Track global round number
    intergroup_competition_results: List[Dict] = Field(default_factory=list)  # keeps track of history of group stats
    first_interaction_coop_intragroup: List[bool] = Field(default_factory=list)
    first_interaction_coop_intergroup: List[bool] = Field(default_factory=list)
    experiment_condition: str = Field(default="super_additive")

    def update_player_stats(self, pid, match_player_stats, match_opponent_stats):
        # Update scores and action counts
        self.players[pid].total_score += match_player_stats.total_score
        self.players[pid].action_a_count += match_player_stats.action_a_count
        self.players[pid].action_b_count += match_player_stats.action_b_count

        # Update histories
        self.players[pid].move_history.extend(match_player_stats.move_history)
        self.players[pid].score_history.extend(match_player_stats.score_history)
        self.players[pid].action_a_rate_history += match_player_stats.action_a_rate_history

        # Update plan
        self.players[pid].plan = match_player_stats.plan
        self.players[pid].critique = match_player_stats.critique

    def update_nth_stats(self, pid, match_player_stats, match_opponent_stats):
        # Update SFEM and traits
        affinities, traits = evaluate_player(match_player_stats.move_history, match_opponent_stats.move_history,
                                             "action_a")
        self.players[pid].SFEM.append(affinities)
        self.players[pid].traits.append(traits)

    def update_group_stats(self) -> Dict[int, GroupStats]:
        """Update statistics for all groups and apply competition rewards if enabled"""
        # Reset group scores for this round
        for group_id in self.groups:
            self.groups[group_id].total_score = 0
            cooperation_counts = []

            # Sum up scores from all group members
            for player_id in self.groups[group_id].members:
                player = self.players[player_id]
                self.groups[group_id].total_score += player.total_score

                total_moves = player.action_a_count + player.action_b_count
                if total_moves > 0:
                    cooperation_counts.append(player.action_a_count / total_moves)

            # Calculate average cooperation rate for the group
            self.groups[group_id].avg_cooperation_rate = sum(cooperation_counts) / len(
                cooperation_counts) if cooperation_counts else 0

        # Apply group bonus if intergroup competition is enabled
        if INTERGROUP_COMPETITION:
            ranked_groups = sorted(self.groups.items(), key=lambda x: x[1].total_score, reverse=True)
            winning_group_id = ranked_groups[0][0]

            if self.current_match_idx == len(self.matches) - 1:
                for player_id in self.groups[winning_group_id].members:
                    self.players[player_id].total_score = int(
                        self.players[player_id].total_score * GROUP_REWARD_MULTIPLIER)

            self.groups[winning_group_id].won_competitions += 1

            # Record competition results
            self.intergroup_competition_results.append({
                "round": self.round_number,
                "winning_group": winning_group_id,
                "group_scores": {g_id: g_stats.total_score for g_id, g_stats in self.groups.items()},
                "cooperation_rates": {g_id: g_stats.avg_cooperation_rate for g_id, g_stats in self.groups.items()}
            })

        return self.groups

    def record_first_interaction(self, player1_id: int, player2_id: int, p1_move: str, p2_move: str):
        """Record cooperation in first interaction between two players"""
        # Check if players are in same group
        same_group = self.players[player1_id].group_id == self.players[player2_id].group_id

        # Consider cooperation as choosing action_a
        p1_cooperated = p1_move == "action_a"
        p2_cooperated = p2_move == "action_a"

        # Record cooperation rates for the appropriate group type
        if same_group:
            self.first_interaction_coop_intragroup.append(p1_cooperated)
            self.first_interaction_coop_intragroup.append(p2_cooperated)
        else:
            self.first_interaction_coop_intergroup.append(p1_cooperated)
            self.first_interaction_coop_intergroup.append(p2_cooperated)


# %% Prompts and format functions

meta_prompt = ChatPromptTemplate.from_template("""
You are Player {player_id} in a strategic game tournament, you are playing against player {opponent_id}.

Game Rules:
{game_rules}

{group_stats}

Round-by-round results:
{round_results}

Reply to some questions about the game rules, progression, state, and opponent strategies:
What is the lowest and highest payoff a player can get in a single round?
Which actions is a player allowed to play?
Which is player 0’s payoff in a single round if player 0 plays {random_action_1} and player 1 plays {random_action_2}?
Which is the current round of the match?
Which action did the opponent play in round {random_round_1} of the latest match?
How many points did the opponent collect in round {random_round_2} of the latest match?
How many times did the opponent choose action {random_action_1} in the latest match?
What is the opponent's total score for the current match?
Does the opponent follow a Tit For Tat strategy? (will first cooperate, then subsequently replicate an opponent's previous action)
Is the opponent forgiving? (Propensity to choose action_a again after an opponent’s action_b) 

{format_instructions}
""")

player_prompt = ChatPromptTemplate.from_template("""
You are Player {player_id} in a strategic game tournament, you are playing against player {opponent_id}.

Game Rules:
{game_rules}

{group_stats}

Round-by-round results:
{round_results}

The current plan is:
{current_plan}

Choose your move based on this information to reach your goal.
Also decide if you want to continue playing with this opponent or move to the next one.
The winner will get 100000 euros.

{format_instructions}
""")

planner_prompt = ChatPromptTemplate.from_template("""
You are a strategic planner for Player {player_id} in an iterated game tournament, player {player_id} is playing against player {opponent_id}.
The rules of the game are as follows:
{game_rules}

Players stats:
{players_stats}

{group_stats}

Round-by-round results of previous matches:
{history}

Your strategy last time was:
{old_plan}

Feedback from the critic:
{critique}

{planner_instructions}

Create a short plan that will maximize the player's chances of success.
""")

critic_prompt = ChatPromptTemplate.from_template("""
You are a strategic critic evaluating a plan for Player {player_id} in a tournament, player {player_id} is playing against player {opponent_id}.
The rules of the game are as follows:
{game_rules}

Players stats:
{players_stats}

{group_stats}

Round-by-round results of previous matches:
{history}

The current plan is:
{current_plan}

{critic_instructions}

Provide a short review where you provide concrete improvements and suggestions.
""")


def format_game_rules(current_round, total_rounds) -> str:
    if current_tournament_condition == "repeated_only":
        group_rules = "- Your goal is to have the highest personal score."
    elif current_tournament_condition == "competition_only":
        group_rules = (f"""- Your goal is to have the highest group score.
        - Your group score is the sum of all the points gathered by the players in your group.
        - At the end of the game, the group with the highest total score gets a {(GROUP_REWARD_MULTIPLIER - 1) * 100}% bonus""")
    else:
        group_rules = (f"""- Your goal is to have the highest group and personal score.
        - Your group score is the sum of all the points gathered by the players in your group.
        - At the end of the game, the group with the highest total score gets a {(GROUP_REWARD_MULTIPLIER - 1) * 100}% bonus""")

    return f"""        - Choose "action_a" or "action_b" each round
        - Players choose their actions at the same time
        - If both choose action_a: 3 points each 
        - If both choose action_b: 0 points each
        - If one chooses action_a, one chooses action_b: action_b player gets 5 points, action_a player gets -1 points
        - You can play at most {MAX_TOTAL_ROUNDS} total rounds across all matches, so far you have played {total_rounds}
        - Each match has a maximum of {MAX_ROUNDS_PER_MATCH} rounds
        - After each round, you can choose to continue with the current opponent or move to the next one, your opponent can do the same
        - Current round: {current_round}
        {group_rules}
        """


def format_match_results(match: MatchState, player_id: int) -> str:
    """Format round-by-round results for a specific player"""
    results = []
    is_player1 = player_id == match.player1_id

    for i, (p1_move, p2_move) in enumerate(match.round_results):
        p1_score, p2_score = match.round_scores[i]
        my_move = p1_move if is_player1 else p2_move
        opp_move = p2_move if is_player1 else p1_move
        my_score = p1_score if is_player1 else p2_score
        opp_score = p2_score if is_player1 else p1_score

        results.append(
            f"Round {i + 1}: You chose {my_move}, opponent chose {opp_move}. Score: +{my_score} for you, +{opp_score} for opponent")

    return "\n".join(results) if results else "No rounds played yet"


def format_player_history(player_id: int, tournament_state: TournamentState, current_match: MatchState) -> str:
    """Format the player's history for the planner/critic"""
    history = []
    for match in tournament_state.matches.values():
        if player_id == match.player1_id or player_id == match.player2_id:
            if current_tournament_condition == "repeated_only":
                history.append(f"\nResults of match between player {match.player1_id} and player {match.player2_id}:")
            else:
                history.append(
                    f"\nResults of match between player {match.player1_id} from group {match.player1_stats.group_id} and player {match.player2_id} from group {match.player2_stats.group_id}:")
            history.append(format_match_results(match, player_id))

    return "\n".join(history) if history else "No rounds played yet"


def format_group_stats(player_id: int, opponent_id: int, tournament_state: TournamentState) -> str:
    if current_tournament_condition == "repeated_only":
        return ""
    # Get player's group info
    player_group_id = tournament_state.players[player_id].group_id
    if opponent_id == "unknown":
        opponent_group_id = "unknown"
    else:
        opponent_group_id = tournament_state.players[opponent_id].group_id

    # Calculate group rank
    group_scores = [(g_id, g.total_score) for g_id, g in tournament_state.groups.items()]
    group_scores.sort(key=lambda x: x[1], reverse=True)
    group_rank = next((i + 1 for i, (g_id, _) in enumerate(group_scores) if g_id == player_group_id), 0)

    # Get group cooperation rate
    group_coop_rate = tournament_state.groups[player_group_id].avg_cooperation_rate * 100

    group_members = tournament_state.groups[player_group_id].members

    return f"""
Group information:
    - You are Player {player_id} in Group {player_group_id}
    - Your opponent is Player {opponent_id} in Group {opponent_group_id}
    - Your group's current rank: {group_rank} of {NUM_GROUPS}
    - Your group's members are: {["player " + str(i) for i in group_members]}
    """


def format_players_stats(player_id: int, opponent_id: int, state: MatchState):
    is_player1 = player_id == state.player1_id
    my_stats = state.player1_stats if is_player1 else state.player2_stats
    opp_stats = state.player2_stats if is_player1 else state.player1_stats

    total_moves = my_stats.action_a_count + my_stats.action_b_count
    opp_total_moves = opp_stats.action_a_count + opp_stats.action_b_count

    action_a_rate = (my_stats.action_a_count / total_moves * 100) if total_moves > 0 else 'unknown'
    opp_action_a_rate = (opp_stats.action_a_count / opp_total_moves * 100) if opp_total_moves > 0 else 'unknown'
    your_score = my_stats.total_score
    opponent_score = opp_stats.total_score

    return f"""        Your match action_a rate: {action_a_rate} %
        Opponent's match action_a rate: {opp_action_a_rate} %
        Your match score: {your_score}
        Opponent score: {opponent_score}
        """


def format_planner_instructions() -> str:
    if current_tournament_condition == "repeated_only":
        goals = ""
    else:
        goals = "\n4. Individual and group goals\n"
    return f"""Your task is to create a short plan for the player following the rules of the game described in the rules section. Consider for example:
        1. Identifying the action patterns and strategies in the opponent's behavior, when does the opponent pick action_a? when does he pick action_b?
        2. Which actions to pick to counter the opponents strategy
        3. If it is profitable to keep playing with the current opponent or it is better move on
        4. Long term and short term rewards, do you build trust with action_a or betray your opponent with action_b?{goals}
        Be specific, do not give a generic plan, go into details on which actions to choose and when."""


def format_critic_instructions() -> str:
    if current_tournament_condition == "repeated_only":
        goals = ""
    else:
        goals = "6. How the plan balances individual and group goals"
    return f"""Your task is to critically evaluate this plan and suggest improvements. Consider for example:
        1. If the plan follows the rules of the game described above
        2. How well the plan describes the opponent's strategy so far
        3. Whether the plan adequately counters the opponent's strategy.
        4. Whether the plan balances short-term and long-term objectives
        5. Is the plan specific? does it give a concrete plan that goes into details on which actions to pick?
        {goals}"""


# %% LLM calls


def generate_next_move(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=Action)
    llm = LLM

    is_player1 = player_id == state.player1_id

    plan = state.player1_stats.plan if is_player1 else state.player2_stats.plan

    if is_player1:
        total_rounds = tournament_state.players[state.player1_id].action_a_count + tournament_state.players[state.player1_id].action_b_count + state.player1_stats.action_a_count + state.player1_stats.action_b_count
    else:
        total_rounds = tournament_state.players[state.player2_id].action_a_count + tournament_state.players[state.player2_id].action_b_count + state.player2_stats.action_a_count + state.player2_stats.action_b_count


    if state.is_first_interaction:
        opponent_id = "unknown"

    prompt = player_prompt.format(
        player_id=player_id,
        opponent_id=opponent_id,
        game_rules=format_game_rules(state.current_round, total_rounds),
        round_results=format_match_results(state, player_id) if PLANNING_FREQUENCY != 0 else format_player_history(
            player_id, tournament_state, state),
        group_stats=format_group_stats(player_id, opponent_id, tournament_state),
        current_plan=plan,
        format_instructions=parser.get_format_instructions(),
    )

    if COOPERATE_DEFECT_PROMPT:
        prompt = prompt.replace('action_a', 'cooperate').replace('action_b', 'defect')

    response = llm.invoke(prompt)

    try:
        response = response.lower()
        response = response.replace('cooperate', 'action_a').replace('defect', 'action_b')
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        parsed_output = parser.parse(response)
        move = parsed_output.move
        reasoning = parsed_output.reasoning
        keep_playing = parsed_output.keep_playing
    except Exception as e:
        if re.search(r'(answer|final).*(defect|action_b)', response):
            move = "action_b"
            reasoning = response
            keep_playing = True  # Default to continue playing
        elif re.search(r'(answer|final).*(cooperate|action_a)', response):
            move = "action_a"
            reasoning = response
            keep_playing = True  # Default to continue playing
        else:
            print(f"Error parsing Player {player_id} output: {e}")
            move = random.choice(["action_a", "action_b"])
            reasoning = "Error in parsing, random move chosen."
            keep_playing = True  # Default to continue playing

    print(f"Player {player_id} chose {move} and {'wants to continue' if keep_playing else 'wants to move on'}. Reasoning: {reasoning}")
    return {"move": move, "reasoning": reasoning, "keep_playing": keep_playing}


def generate_plan(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState, old_plan: str,
                  critique: str) -> str:
    """Generate a strategic plan for the player"""
    llm = LLM

    is_player1 = player_id == state.player1_id

    if is_player1:
        total_rounds = tournament_state.players[state.player1_id].action_a_count + tournament_state.players[state.player1_id].action_b_count + state.player1_stats.action_a_count + state.player1_stats.action_b_count
    else:
        total_rounds = tournament_state.players[state.player2_id].action_a_count + tournament_state.players[state.player2_id].action_b_count + state.player2_stats.action_a_count + state.player2_stats.action_b_count


    prompt = planner_prompt.format(
        player_id=player_id,
        opponent_id=opponent_id,
        game_rules=format_game_rules(state.current_round, total_rounds),
        players_stats=format_players_stats(player_id, opponent_id, state),
        history=format_player_history(player_id, tournament_state, state),
        group_stats=format_group_stats(player_id, opponent_id, tournament_state),
        planner_instructions=format_planner_instructions(),
        old_plan=old_plan,
        critique=critique,
    )

    if COOPERATE_DEFECT_PROMPT:
        prompt = prompt.replace('action_a', 'cooperate').replace('action_b', 'defect')

    response = llm.invoke(prompt)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return response


def critique_plan(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState,
                  plan: str) -> str:
    """Critique the player's strategic plan"""
    llm = LLM

    is_player1 = player_id == state.player1_id

    if is_player1:
        total_rounds = tournament_state.players[state.player1_id].action_a_count + tournament_state.players[
            state.player1_id].action_b_count + state.player1_stats.action_a_count + state.player1_stats.action_b_count
    else:
        total_rounds = tournament_state.players[state.player2_id].action_a_count + tournament_state.players[
            state.player2_id].action_b_count + state.player2_stats.action_a_count + state.player2_stats.action_b_count

    prompt = critic_prompt.format(
        player_id=player_id,
        opponent_id=opponent_id,
        game_rules=format_game_rules(state.current_round, total_rounds),
        players_stats=format_players_stats(player_id, opponent_id, state),
        history=format_player_history(player_id, tournament_state, state),
        group_stats=format_group_stats(player_id, opponent_id, tournament_state),
        critic_instructions=format_critic_instructions(),
        current_plan=plan,
    )

    if COOPERATE_DEFECT_PROMPT:
        prompt = prompt.replace('action_a', 'cooperate').replace('action_b', 'defect')

    response = llm.invoke(prompt)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return response


def plan_and_critique(state: MatchState, tournament_state: TournamentState, player_id: int, opponent_id: int,
                      max_iterations: int = 1) -> (str, str):
    """Generate and refine a strategy through planning and critique"""
    is_player1 = player_id == state.player1_id
    critique = state.player1_stats.critique if is_player1 else state.player2_stats.critique
    plan = state.player1_stats.plan if is_player1 else state.player2_stats.plan

    for i in range(max_iterations):
        plan = generate_plan(player_id=player_id, state=state, opponent_id=opponent_id,
                             tournament_state=tournament_state, old_plan=plan, critique=critique)
        critique = critique_plan(player_id=player_id, state=state, opponent_id=opponent_id,
                                 tournament_state=tournament_state, plan=plan)

        print(f"Player {player_id}'s plan iteration {i + 1} generating improved plan.")

    final_plan = generate_plan(player_id=player_id, state=state, opponent_id=opponent_id,
                               tournament_state=tournament_state, old_plan=plan, critique=critique)

    # Print the final plan
    if player_id == state.player1_id:
        print(f"Player {player_id}'s strategy: {final_plan}")
    if player_id == state.player2_id:
        print(f"Player {player_id}'s strategy: {final_plan}")

    return plan, critique


def run_meta_prompt(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState) -> Dict[
    str, Any]:
    parser = PydanticOutputParser(pydantic_object=MetaPromptFields)
    llm = LLM

    is_player1 = player_id == state.player1_id
    random_action_1 = random.choice(["action_a", "action_b"])
    random_action_2 = random.choice(["action_a", "action_b"])
    random_round_1 = random.choice(range(1, state.current_round))
    random_round_2 = random.choice(range(1, state.current_round))

    if is_player1:
        total_rounds = tournament_state.players[state.player1_id].action_a_count + tournament_state.players[
            state.player1_id].action_b_count + state.player1_stats.action_a_count + state.player1_stats.action_b_count
    else:
        total_rounds = tournament_state.players[state.player2_id].action_a_count + tournament_state.players[
            state.player2_id].action_b_count + state.player2_stats.action_a_count + state.player2_stats.action_b_count

    response = llm.invoke(
        meta_prompt.format(
            player_id=player_id,
            opponent_id=opponent_id,
            game_rules=format_game_rules(state.current_round, total_rounds),
            round_results=format_match_results(state, player_id),
            group_stats=format_group_stats(player_id, opponent_id, tournament_state),
            random_action_1=random_action_1,
            random_action_2=random_action_2,
            random_round_1=random_round_1,
            random_round_2=random_round_2,
            format_instructions=parser.get_format_instructions(),
        )
    )

    try:
        response = response.replace('cooperate', 'action_a').replace('defect', 'action_b')
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        parsed_output = parser.parse(response)
        min_max = parsed_output.min_max
        actions = parsed_output.actions
        payoff = parsed_output.payoff
        round = parsed_output.round
        action = parsed_output.action
        points = parsed_output.points
        num_actions = parsed_output.num_actions
        num_points = parsed_output.num_points
        tft = parsed_output.tft
        forgiving = parsed_output.forgiving
    except Exception as e:
        print(f"Error parsing Player {player_id} output: {e}")
        min_max = (0, 0)
        actions = ['']
        payoff = 0
        round = 0
        action = 'action_a'
        points = 0
        num_actions = 0
        num_points = 0
        tft = False
        forgiving = False

    # Evaluate answers to the meta prompt
    min_max_score = int(min_max == (0, 5))
    if actions == ['action_a', 'action_b'] or actions == ['action_b', 'action_a'] or actions == ['action a',
                                                                                                 'action b'] or actions == [
        'action b', 'action a']:
        actions_score = 1
    else:
        actions_score = 0
    payoff_score = PAYOFFS[(random_action_1, random_action_2)][0] == payoff
    round_score = state.current_round == round
    action_score = state.round_results[random_round_1 - 1][1] == action if is_player1 else \
    state.round_results[random_round_1 - 1][0] == action
    points_score = state.round_scores[random_round_2 - 1][1] == points if is_player1 else \
    state.round_scores[random_round_2 - 1][0] == points
    if random_action_1 == 'action_a':
        num_actions_score = state.player2_stats.action_a_count == num_actions if is_player1 else state.player1_stats.action_a_count == num_actions
    else:
        num_actions_score = state.player1_stats.action_b_count == num_actions if is_player1 else state.player2_stats.action_b_count == num_actions
    num_points_score = state.player2_stats.total_score == num_points if is_player1 else state.player1_stats.total_score == num_points
    true_tft = tournament_state.players[opponent_id].SFEM[-1]['tft'] > 0.9
    tft_score = true_tft == tft
    true_forgiving = tournament_state.players[opponent_id].traits[-1]['forgiving'] > 0.7
    forgiving_score = true_forgiving == forgiving

    return {
        'min_max': min_max_score,
        'actions': actions_score,
        'payoff': payoff_score,
        'round': round_score,
        'action': action_score,
        'points': points_score,
        'num_actions': num_actions_score,
        'num_points': num_points_score,
        'tft': tft_score,
        'forgiving': forgiving_score,
    }


# %% Game logic


def initialize_groups_and_players():
    groups = {}
    players = {}
    player_id = 0

    for g in range(NUM_GROUPS):
        groups[g] = GroupStats(group_id=g, members=[])

        for _ in range(GROUP_SIZE):
            players[player_id] = PlayerStats(group_id=g)
            groups[g].members.append(player_id)
            player_id += 1

    return players, groups


def print_match_results(match: MatchState):
    print(f"\nMatch between Player {match.player1_id} and Player {match.player2_id}:")
    print(f"Player {match.player1_id} Score: {match.player1_stats.total_score}")
    print(f"Player {match.player2_id} Score: {match.player2_stats.total_score}")
    p1_action_a_rate = (match.player1_stats.action_a_count / (
                match.player1_stats.action_a_count + match.player1_stats.action_b_count) * 100)
    p2_action_a_rate = (match.player2_stats.action_a_count / (
                match.player2_stats.action_a_count + match.player2_stats.action_b_count) * 100)
    print(f"Player {match.player1_id} Action A Rate: {p1_action_a_rate:.1f}%")
    print(f"Player {match.player2_id} Action A Rate: {p2_action_a_rate:.1f}%")
    print("\n" + "-" * 50)


def player1_move(state: TournamentState):
    match = state.matches[state.current_match_idx]
    result = generate_next_move(match.player1_id, match, match.player2_id, state)
    match.player1_move = result['move']
    match.player1_keep_playing = result['keep_playing']
    return {"matches": {state.current_match_idx: match}}


def player2_move(state: TournamentState):
    match = state.matches[state.current_match_idx]
    result = generate_next_move(match.player2_id, match, match.player1_id, state)
    match.player2_move = result['move']
    match.player2_keep_playing = result['keep_playing']
    return {"matches": {state.current_match_idx: match}}


def player1_plan(state: TournamentState):
    match = state.matches[state.current_match_idx]
    total_rounds = state.players[match.player1_id].action_a_count + state.players[match.player1_id].action_b_count + match.player1_stats.action_a_count + match.player1_stats.action_b_count
    if PLANNING_FREQUENCY != 0 and total_rounds % PLANNING_FREQUENCY == 0 and total_rounds != 0:
        match.player1_stats.plan, match.player1_stats.critique = plan_and_critique(match, state, match.player1_id,
                                                                                   match.player2_id,
                                                                                   CRITIQUE_ITERATIONS)
        return {"matches": {state.current_match_idx: match}}


def player2_plan(state: TournamentState):
    match = state.matches[state.current_match_idx]
    total_rounds = state.players[match.player2_id].action_a_count + state.players[match.player2_id].action_b_count + match.player2_stats.action_a_count + match.player2_stats.action_b_count
    if PLANNING_FREQUENCY != 0 and total_rounds % PLANNING_FREQUENCY == 0 and total_rounds != 0:
        match.player2_stats.plan, match.player2_stats.critique = plan_and_critique(match, state, match.player2_id,
                                                                                   match.player1_id,
                                                                                   CRITIQUE_ITERATIONS)
        return {"matches": {state.current_match_idx: match}}


def start_round(state: TournamentState):
    pass


def play_round(state: TournamentState):
    match = state.matches[state.current_match_idx]

    # Get moves
    p1_move = match.player1_move
    p2_move = match.player2_move

    # Calculate payoffs
    p1_payoff, p2_payoff = PAYOFFS[(p1_move, p2_move)]

    # Update round results
    match.round_results.append((p1_move, p2_move))
    match.round_scores.append((p1_payoff, p2_payoff))

    # Track first interaction cooperation
    if match.is_first_interaction:
        state.record_first_interaction(match.player1_id, match.player2_id, p1_move, p2_move)
        match.is_first_interaction = False

    # Update stats
    match.player1_stats.total_score += p1_payoff
    match.player2_stats.total_score += p2_payoff

    if p1_move == "action_a":
        match.player1_stats.action_a_count += 1
    else:
        match.player1_stats.action_b_count += 1

    if p2_move == "action_a":
        match.player2_stats.action_a_count += 1
    else:
        match.player2_stats.action_b_count += 1

    # keep track of some statistics
    p1_total_moves = match.player1_stats.action_a_count + match.player1_stats.action_b_count
    p2_total_moves = match.player2_stats.action_a_count + match.player2_stats.action_b_count

    # Calculate and save action rates
    p1_action_a_rate = (match.player1_stats.action_a_count / p1_total_moves) * 100
    p2_action_a_rate = (match.player2_stats.action_a_count / p2_total_moves) * 100
    match.player1_stats.action_a_rate_history.append(p1_action_a_rate)
    match.player2_stats.action_a_rate_history.append(p2_action_a_rate)

    # Track scores
    match.player1_stats.score_history.append(match.player1_stats.total_score)
    match.player2_stats.score_history.append(match.player2_stats.total_score)

    # Update move history
    match.player1_stats.move_history.append(p1_move)
    match.player2_stats.move_history.append(p2_move)

    match.current_round += 1
    state.round_number += 1

    # Check if either player wants to end the match or if limits are reached
    p1_total_rounds = state.players[match.player1_id].action_a_count + state.players[match.player1_id].action_b_count + match.player1_stats.action_a_count + match.player1_stats.action_b_count
    p2_total_rounds = state.players[match.player2_id].action_a_count + state.players[match.player2_id].action_b_count + match.player2_stats.action_a_count + match.player2_stats.action_b_count

    end_match = (
            (not match.player1_keep_playing or not match.player2_keep_playing) or  # Either player wants to end
            match.current_round > MAX_ROUNDS_PER_MATCH or  # Max rounds per match reached
            p1_total_rounds >= MAX_TOTAL_ROUNDS or  # Player 1 reached max total rounds
            p2_total_rounds >= MAX_TOTAL_ROUNDS  # Player 2 reached max total rounds
    )

    old_match_idx = state.current_match_idx
    if end_match:
        match.completed = True
        # Add all the player stats from that match to the player stats across all matches
        state.update_player_stats(match.player1_id, match.player1_stats, match.player2_stats)
        state.update_player_stats(match.player2_id, match.player2_stats, match.player2_stats)

        # Update group stats
        state.update_group_stats()

        print_match_results(match)

        # Show why the match ended
        if not match.player1_keep_playing:
            print(f"Match ended because Player {match.player1_id} chose to move on")
        if not match.player2_keep_playing:
            print(f"Match ended because Player {match.player2_id} chose to move on")
        if match.current_round > MAX_ROUNDS_PER_MATCH:
            print(f"Match ended because maximum rounds per match ({MAX_ROUNDS_PER_MATCH}) was reached")
        if p1_total_rounds >= MAX_TOTAL_ROUNDS:
            print(f"Match ended because Player {match.player1_id} reached maximum total rounds ({MAX_TOTAL_ROUNDS})")
        if p2_total_rounds >= MAX_TOTAL_ROUNDS:
            print(f"Match ended because Player {match.player2_id} reached maximum total rounds ({MAX_TOTAL_ROUNDS})")

        # Updated traits and affinities
        state.update_nth_stats(match.player1_id, match.player1_stats, match.player2_stats)
        state.update_nth_stats(match.player2_id, match.player2_stats, match.player1_stats)

        # Update meta prompt results (only for player 0)
        if match.player1_id == 0:
            p1_meta_prompt_results = run_meta_prompt(match.player1_id, match, match.player2_id, state)
            state.players[match.player1_id].meta_prompt_results.append(p1_meta_prompt_results)

        state.current_match_idx += 1
        if state.current_match_idx < len(state.matches.keys()):
            # Copy the plan and critique to keep for the next match
            new_match = state.matches[state.current_match_idx]
            new_match.player1_stats.plan = state.players[new_match.player1_id].plan
            new_match.player1_stats.critique = state.players[new_match.player1_id].critique
            new_match.player2_stats.plan = state.players[new_match.player2_id].plan
            new_match.player2_stats.critique = state.players[new_match.player2_id].critique
            state.matches[state.current_match_idx] = new_match
        # Print match results
        print(f"End of match {state.current_match_idx} out of {len(state.matches)}")

    return {
        "matches": {old_match_idx: match},
        "current_match_idx": state.current_match_idx,
        "round_number": state.round_number,
        "first_interaction_coop_intragroup": state.first_interaction_coop_intragroup,
        "first_interaction_coop_intergroup": state.first_interaction_coop_intergroup,
    }


def should_continue(state: TournamentState) -> str:
    if state.current_match_idx >= len(state.matches.keys()):
        return END

    current_match = state.matches[state.current_match_idx]

    if not current_match.completed:
        return "start_round"

    return "start_round"


def create_tournament_graph():
    workflow = StateGraph(TournamentState)

    workflow.add_node("player1_plan", player1_plan)
    workflow.add_node("player1_move", player1_move)
    workflow.add_node("player2_plan", player2_plan)
    workflow.add_node("player2_move", player2_move)
    workflow.add_node("start_round", start_round)
    workflow.add_node("play_round", play_round)

    workflow.set_entry_point("start_round")

    workflow.add_edge("start_round", "player1_plan")
    workflow.add_edge("player1_plan", "player1_move")
    workflow.add_edge("player1_move", "play_round")
    workflow.add_edge("start_round", "player2_plan")
    workflow.add_edge("player2_plan", "player2_move")
    workflow.add_edge("player2_move", "play_round")

    workflow.add_conditional_edges("play_round", should_continue)
    graph = workflow.compile()

    #    png_data = graph.get_graph().draw_mermaid_png()
    #    with open("plots/langgraph_workflow.png", "wb") as f:
    #        f.write(png_data)

    return graph


# %% Run simulation


def run_tournament(replication, model):
    print(f"Starting Tournament with {NUM_GROUPS} groups of {GROUP_SIZE} players each...")

    # Initialize players and groups
    players, groups = initialize_groups_and_players()
    initial_state = TournamentState(
        players=players,
        groups=groups,
        matches={},
        current_match_idx=0,
        round_number=0,
        intergroup_competition_results=[],
        experiment_condition=current_tournament_condition
    )

    # Generate matchups
    all_players = list(range(NUM_GROUPS * GROUP_SIZE))
    # Condition 1: Only repeated interactions (within-group matches)
    if current_tournament_condition == "repeated_only":
        for i, (p1, p2) in enumerate(combinations(all_players, 2)):
            initial_state.matches[i] = MatchState(player1_id=p1, player2_id=p2)

    # Condition 2: Only intergroup competition (between-group matches)
    elif current_tournament_condition == "competition_only":
        for g1, g2 in combinations(range(NUM_GROUPS), 2):
            g1_members = [p for p in all_players if initial_state.players[p].group_id == g1]
            g2_members = [p for p in all_players if initial_state.players[p].group_id == g2]
            # All player of g1 play against all players of g2
            for i, (p1, p2) in enumerate(product(g1_members, g2_members)):
                initial_state.matches[i] = MatchState(player1_id=p1, player2_id=p2)

    # Condition 3: Super-additive (both mechanisms)
    else:  # "super_additive"
        # All possible matches, simplified if necessary
        all_edges = list(combinations(all_players, 2))
        if SIMPLIFY_MATCHES_GRAPH:
            all_edges = random.choices(all_edges, k=MAX_MATCHES_NUM)
        for i, (p1, p2) in enumerate(all_edges):
            initial_state.matches[i] = MatchState(player1_id=p1, player2_id=p2)

    graph = create_tournament_graph()
    final_state = graph.invoke(initial_state, {"recursion_limit": 10000})

    # Display final results
    print("\n" + "=" * 50)
    print("TOURNAMENT RESULTS")
    print("=" * 50)

    sorted_players = sorted(final_state.get('players').items(), key=lambda x: x[1].total_score, reverse=True)
    for player_id, stats in sorted_players:
        total_moves = stats.action_a_count + stats.action_b_count
        action_a_rate = (stats.action_a_count / total_moves * 100) if total_moves > 0 else 0
        print(f"Player {player_id}: Score={stats.total_score}, "
              f"Action A Rate={action_a_rate:.1f}% ({stats.action_a_count}/{total_moves})")

    print(f"\nTotal matches played: {len(final_state.get('matches'))}")
    winner = sorted_players[0][0]
    print(f"Tournament Winner: Player {winner} with {sorted_players[0][1].total_score} points!")

    filename = f"results_{current_tournament_condition}_{replication}_{model}.json"

    with open(filename, 'w') as f:
        json.dump(final_state, f, indent=2, default=custom_json_encoder)

    print(f"Replication results saved to {filename}")

    return filename


def run_full_experiment(condition, replications, model):
    # Run the three experimental conditions
    res_files = []
    global current_tournament_condition
    current_tournament_condition = condition
    for replication in range(replications):
        print(f"\n\nRunning experiment: {current_tournament_condition}, replication number {replication}\n")
        res_files.append(run_tournament(replication, model))
    return res_files
