# SFEM metrics
def allc(opponent_moves):
    return ['cooperate'] * len(opponent_moves)


def alld(opponent_moves):
    return ['defect'] * len(opponent_moves)


def tft(opponent_moves):
    if not opponent_moves:
        return []
    moves = ['cooperate']
    for t in range(1, len(opponent_moves)):
        moves.append(opponent_moves[t-1])
    return moves


def grim(opponent_moves):
    moves = []
    for t in range(len(opponent_moves)):
        if any(m == 'defect' for m in opponent_moves[:t]):
            moves.append('defect')
        else:
            moves.append('cooperate')
    return moves


def pavlov(opponent_moves):
    if not opponent_moves:
        return []
    moves = ['cooperate']
    for t in range(1, len(opponent_moves)):
        if moves[t-1] == opponent_moves[t-1]:
            moves.append('cooperate')
        else:
            moves.append('defect')
    return moves


def affinity(player_moves, strategy_moves):
    if not player_moves:
        return 0
    matches = sum(p == s for p,s in zip(player_moves, strategy_moves))
    return matches / len(player_moves)


# Nicer than humans metrics
def is_nice(player_moves, opponent_moves):
    for t in range(len(player_moves)):
        if player_moves[t] == 'defect':
            if all(m == 'cooperate' for m in opponent_moves[:t]) and opponent_moves[t] == 'cooperate':
                return 0
    return 1


def forgiving_metric(player_moves, opponent_moves):
    N = len(player_moves)
    if N < 2:
        return 0
    forgiven = 0
    penalties = 0
    opponent_defections = sum(m == 'defect' for m in opponent_moves)
    for t in range(N-1):
        if opponent_moves[t] == 'defect':
            if player_moves[t+1] == 'cooperate':
                forgiven += 1
            elif opponent_moves[t+1] == 'cooperate' and player_moves[t+1] == 'defect':
                penalties += 1
    denominator = opponent_defections + penalties
    if denominator == 0:
        return 0
    else:
        return forgiven / denominator


def retaliatory_metric(player_moves, opponent_moves):
    N = len(player_moves)
    if N < 2:
        return 0
    provocations = 0
    reactions = 0
    for t in range(1, N):
        if opponent_moves[t] == 'defect' and player_moves[t-1] == 'cooperate':
            provocations += 1
            if player_moves[t] == 'defect':
                reactions += 1
    if provocations == 0:
        return 0
    else:
        return reactions / provocations


def troublemaking_metric(player_moves, opponent_moves):
    N = len(player_moves)
    if N == 0:
        return 0
    occasions = 1  # for t=0
    uncalled = 0
    if player_moves[0] == 'defect':
        uncalled += 1
    for t in range(1, N):
        if opponent_moves[t-1] == 'cooperate':
            occasions += 1
            if player_moves[t] == 'defect':
                uncalled += 1
    if occasions == 0:
        return 0
    else:
        return uncalled / occasions


def emulative_metric(player_moves, opponent_moves):
    N = len(player_moves)
    if N < 2:
        return 0
    mimic = sum(player_moves[t] == opponent_moves[t-1] for t in range(1, N))
    return mimic / (N - 1)


def evaluate_player(player_moves, opponent_moves, cooperate_action_name):

    for i, action in enumerate(player_moves):
        if action == cooperate_action_name:
            player_moves[i] = "cooperate"
        else:
            player_moves[i] = "defect"

    for i, action in enumerate(opponent_moves):
        if action == cooperate_action_name:
            opponent_moves[i] = "cooperate"
        else:
            opponent_moves[i] = "defect"

    affinities = {
        'allc': affinity(player_moves, allc(opponent_moves)),
        'alld': affinity(player_moves, alld(opponent_moves)),
        'tft': affinity(player_moves, tft(opponent_moves)),
        'grim': affinity(player_moves, grim(opponent_moves)),
        'pavlov': affinity(player_moves, pavlov(opponent_moves)),
    }
    traits = {
        'nice': is_nice(player_moves, opponent_moves),
        'forgiving': forgiving_metric(player_moves, opponent_moves),
        'retaliatory': retaliatory_metric(player_moves, opponent_moves),
        'troublemaking': troublemaking_metric(player_moves, opponent_moves),
        'emulative': emulative_metric(player_moves, opponent_moves),
    }
    return affinities, traits

