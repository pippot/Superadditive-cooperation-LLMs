from IPDv5 import *


class MetaPromptFields(BaseModel):
    min_max: Tuple[int, int] = Field(description="lowest and highest payoff a player can get in a single round")
    actions: List[str] = Field(description="actions is a player allowed to play (use only lower case letters)")
    payoff: int = Field(description="player X’s payoff in a single round if X plays p and Y plays q")
    round: int = Field(description="current round of the game")
    action: Literal["action_a", "action_b"] = Field(description="action player X played in round i")
    points: int = Field(description="points player X collected in round i")
    num_actions: int = Field(description="how many times did player X choose action p")
    num_points: int = Field(description="what is player's X current total score")
    tft: bool = Field(description="whether player X uses a Tit For Tat strategy (will first cooperate, then subsequently replicate an opponent's previous action)")
    forgiving: bool = Field(description="whether player X is forgiving (Propensity to choose action a again after an opponent’s action b)")


meta_prompt = ChatPromptTemplate.from_template("""
You are Player {player_id} in a strategic game tournament.

Game Rules:
{game_rules}

Group information:
{group_stats}

Round-by-round results:
{round_results}

Reply to some questions about the game rules, progression, state, and opponent strategies:
What is the lowest and highest payoff a player can get in a single round?
Which actions is a player allowed to play?
Which is player 0’s payoff in a single round if player 0 plays {random_action_1} and player 1 plays {random_action_2}?
Which is the current round of the match?
Which action did player {opponent_id} play in round {random_round_1}?
How many points did player {opponent_id} collect in round {random_round_2}?
How many times did {opponent_id} choose action {random_action_1}?
What is player {opponent_id}'s total score for the current match?
Does player {opponent_id} uses a Tit For Tat strategy? (will first cooperate, then subsequently replicate an opponent's previous action)
Does player {opponent_id} is forgiving? (Propensity to choose action a again after an opponent’s action b) 


{format_instructions}
""")

def generate_next_move(player_id: int, state: MatchState, opponent_id: int, tournament_state: TournamentState) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=MetaPromptFields)
    llm = LLM

    is_player1 = player_id == state.player1_id

    plan = state.player1_stats.plan if is_player1 else state.player2_stats.plan

    response = llm.invoke(
        player_prompt.format(
            player_id=player_id,
            game_rules=format_game_rules(state.current_round),
            round_results=format_round_results(state, player_id),
            group_stats=format_group_stats(player_id, opponent_id, tournament_state),
            current_plan=plan,
            format_instructions=parser.get_format_instructions(),
            )
    )

    try:
        parsed_output = parser.parse(response)
        move = parsed_output.move
        reasoning = parsed_output.reasoning
    except Exception as e:
        print(f"Error parsing Player {player_id} output: {e}")
        move = random.choice(["action_a", "action_b"])
        reasoning = "Error in parsing, random move chosen."

    print(f"Player {player_id} chose {move}. Reasoning: {reasoning}")
    return {"move": move, "reasoning": reasoning}
