# JSettlers2 Bot - Python Implementation for Catanatron

A comprehensive Python port of the JSettlers2 robot AI for use with the Catanatron Settlers of Catan engine.

## Overview

This package contains a sophisticated AI bot that implements strategic planning, opponent tracking, and trade negotiation capabilities. It's a port of the ~5,500 line Java `SOCRobotBrain` class from JSettlers2, adapted to work with Catanatron's synchronous, turn-based architecture.

### Key Features

- **Strategic Planning**: Uses smart or fast game strategies to plan building priorities
- **Opponent Tracking**: Predicts opponent resources and strategies
- **Trade Negotiation**: Evaluates and creates trade offers
- **Multiple Difficulty Levels**: EASY, MODERATE, HARD, and SMART
- **Comprehensive Game Phase Handling**: Initial placement, main turns, robber movement, discarding, etc.

## Architecture

### Main Components

```
jsettlers2_bot_py/
├── robot_brain.py              # Main bot class (Player implementation)
├── robot_parameters.py         # Configuration and difficulty settings
├── decision/
│   ├── decision_maker.py       # Core planning logic (port of SOCRobotDM)
│   └── building_plan.py        # Build plan and possible piece classes
├── strategies/
│   ├── opening_build.py        # Initial settlement placement
│   ├── robber_strategy.py      # Robber placement logic
│   ├── monopoly_strategy.py    # Monopoly card usage
│   ├── discard_strategy.py     # Resource discarding
│   └── dev_card_strategy.py    # Development card usage
├── tracking/
│   ├── player_tracker.py       # Opponent tracking and prediction
│   └── building_speed_estimate.py  # Build speed calculations
├── trading/
│   └── negotiator.py           # Trade offer creation and evaluation
└── utils/
    ├── game_utils.py           # Game state access helpers
    └── board_utils.py          # Board analysis helpers
```

### Key Differences from Java Version

| Aspect | JSettlers (Java) | This Implementation (Python) |
|--------|------------------|------------------------------|
| **Execution Model** | Thread with message queue | Synchronous `decide()` calls |
| **State Tracking** | Boolean flags (`expectSTART1A`, `waitingFor*`) | Not needed - Catanatron provides state |
| **Complexity** | 5,567 lines, highly stateful | ~1,500 lines core logic |
| **Message Handling** | Giant switch in `run()` loop | Route by game phase in `decide()` |

## Installation

### Prerequisites

```bash
# Ensure you have Catanatron installed
cd /path/to/catanrl/catanatron
pip install -e .
```

### Usage

The bot can be used anywhere Catanatron's `Player` interface is expected:

```python
from catanatron.game import Game
from catanatron.models.player import Color
from jsettlers2_bot_py import JSettlersRobotBrain, RobotDifficulty

# Create bots with different difficulties
smart_bot = JSettlersRobotBrain(Color.RED, difficulty=RobotDifficulty.SMART)
moderate_bot = JSettlersRobotBrain(Color.BLUE, difficulty=RobotDifficulty.MODERATE)
easy_bot = JSettlersRobotBrain(Color.ORANGE, difficulty=RobotDifficulty.EASY)

# Create and play a game
game = Game([smart_bot, moderate_bot, easy_bot, ...])
game.play()

print(f"Winner: {game.winning_color()}")
```

## Difficulty Levels

### EASY
- Uses fast strategy (simple rules)
- Never trades
- Defensive robber placement
- Lower strategic weights

### MODERATE  
- Uses fast strategy
- Trades sometimes
- Balanced robber placement
- Moderate strategic weights

### HARD
- Uses smart strategy (player tracking + ETAs)
- Always tries trading
- Smart robber placement
- Standard strategic weights

### SMART (Default)
- Uses smart strategy
- Always tries trading  
- Aggressive robber placement
- Enhanced strategic weights
- Higher threat response

## Custom Configuration

You can create custom bot configurations:

```python
from jsettlers2_bot_py import JSettlersRobotBrain, RobotParameters

# Create custom parameters
params = RobotParameters(
    strategy_type=1,  # SMART_STRATEGY
    trade_flag=2,  # Always try trading
    adversarial_factor=2.0,  # More aggressive blocking
    threat_multiplier=1.5,  # Respond strongly to threats
)

custom_bot = JSettlersRobotBrain(Color.RED, robot_parameters=params)
```

## Strategy Overview

### Decision Flow

1. **Roll Phase**: Decide whether to play knight before rolling
2. **Planning**: Create/update building plan based on game state
3. **Execution**:
   - Try to build next piece in plan
   - If can't afford: use dev cards (Year of Plenty, Monopoly)
   - If still can't: trade with bank/ports
   - If still can't: trade with players
   - Last resort: buy dev card or end turn

### Building Priority (Smart Strategy)

The bot evaluates all possible pieces and chooses based on:

1. **Cities**: Highest priority if good speedup value
2. **Settlements**: Based on production value, diversity, and expansion potential
3. **Roads**: For longest road (when at 5+ VP)
4. **Dev Cards**: Strategic backup option

### Planning Algorithm

Port of `planStuff()` from JSettlers2:

1. Calculate building speed estimates based on production
2. Update player trackers with win game ETAs
3. Score all possible settlements, cities, and roads
4. Choose favorites based on scores
5. Build prioritized plan considering:
   - Win game ETA comparisons
   - Threat responses
   - Necessary roads for settlements
   - Longest road pursuit

## API Reference

### JSettlersRobotBrain

Main bot class that implements `Player` interface.

```python
class JSettlersRobotBrain(Player):
    def __init__(self, color: Color, 
                 difficulty: Optional[RobotDifficulty] = None,
                 robot_parameters: Optional[RobotParameters] = None)
    
    def decide(self, game, playable_actions: List[Action]) -> Action
    def reset_state(self)  # Called between games
```

### RobotParameters

Configuration for bot behavior.

```python
@dataclass
class RobotParameters:
    strategy_type: int = 1              # 0=FAST, 1=SMART
    trade_flag: int = 1                 # 0=never, 1=sometimes, 2=always
    eta_bonus_factor: float = 0.8
    adversarial_factor: float = 1.5
    leader_adversarial_factor: float = 3.0
    # ... and more
    
    @classmethod
    def create_for_difficulty(cls, difficulty: RobotDifficulty)
```

## Known Limitations & TODOs

### Current Limitations

1. **Simplified Board Analysis**: Some board evaluation functions use placeholders
2. **Port Detection**: Port access not yet fully implemented
3. **Road Pathfinding**: Path calculation for necessary roads is simplified
4. **Trade Acceptance**: Bot doesn't yet respond to incoming trade offers (only creates them)
5. **Scenario Support**: JSettlers scenario-specific logic not ported

### Areas for Enhancement

- [ ] Implement more sophisticated settlement scoring
- [ ] Add proper port detection and bonus calculation
- [ ] Implement A* pathfinding for road planning
- [ ] Add trade offer acceptance/rejection logic
- [ ] Improve production estimation with robber consideration
- [ ] Add support for Catanatron-specific scenarios
- [ ] Optimize decision speed for large game trees

## Testing

Run the included example to test basic functionality:

```bash
python example_usage.py
```

For more comprehensive testing:

```python
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from jsettlers2_bot_py import JSettlersRobotBrain, RobotDifficulty

# Test against random players
bot = JSettlersRobotBrain(Color.RED, RobotDifficulty.SMART)
opponents = [RandomPlayer(c) for c in [Color.BLUE, Color.ORANGE, Color.WHITE]]

wins = 0
for i in range(100):
    game = Game([bot] + opponents, seed=i)
    game.play()
    if game.winning_color() == Color.RED:
        wins += 1

print(f"Bot won {wins}/100 games against random players")
```

## Contributing

This is a port of the JSettlers2 bot, so contributions that improve parity with the Java version or adapt it better to Catanatron are welcome.

### Priority Areas:
1. Board evaluation improvements
2. Trade offer response handling  
3. Testing and benchmarking
4. Performance optimization

## Credits

- **Original JSettlers2 Authors**: Robert S. Thomas, Jeremy D Monin, and contributors
- **Catanatron**: Engine by contributors to the Catanatron project
- **This Port**: Adapted for Catanatron's architecture

## License

Follows the same GPL-3.0 license as JSettlers2.

## References

- [JSettlers2 GitHub](https://github.com/jdmonin/JSettlers2)
- [Catanatron](https://github.com/bcollazo/catanatron)
- Original Java source: `JSettlers2/src/main/java/soc/robot/SOCRobotBrain.java`





