# comp472-project

## Our heuristics:

### e0 (already given)
formula:

### e1
Reasoning:
formula:

### e2
Reasoning:
formula:

## Steps to start the program
1. To play the game with the default parameters, run the ai_wargame.py file using this at your command line:
```
C:\Users\...yourdirectory...> python ai_wargame.py
```

2. To specify game parameters, run the ai_wargame.py file using this at your command line:

```
C:\Users\...yourdirectory...> python ai_wargame.py --max_depth 10 --max_time 10 --max_turns 1000 --game_type attacker
```

Here is the list of game parameters:
| Game Parameter | Description |
| ----------- | ----------- |
| alpha_beta | determines if alpha-beta pruning is turned on or off. Default is True. |
| heuristic | integer between 0 and 2 inclusive that determines which heuristic between e0, e1, and e2 is applied | 
| max_depth | maximum number of depths that the minimax algorithm should explore. |
| min_depth | minimum number of depths that the minimax algorithm should explore. |
| max_time | the maximum time in which the minimax algorithm must return a move |
| max_turns | maximum number of turns allowed for the game. If reached, defender automatically wins. |
| game_type | see below |
```python
                     # attacker vs defender
--game_type manual   # human vs human
--game_type attacker # human vs computer
--game_type defender # computer vs human
--game_type auto     # computer vs computer
```
