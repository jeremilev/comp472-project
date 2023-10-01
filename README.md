# comp472-project

## Quick notes
- Only Human vs Human is fully supported at the moment.
- Human vs Human is the default game parameter.

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
| max_depth | maximum number of depths that the minimax algorithm should explore. |
| min_depth | minimum number of depths that the minimax algorithm should explore. |
| max_time | the maximum time in which the minimax algorithm must return a move |
| max_turns | maximum number of turns allowed for the game. If reached, defender automatically wins. |
| ... | <b>More to come with demo2 </b>|
| game_type | see below |
```python
                     # attacker vs defender
--game_type manual   # human vs human
--game_type attacker #human vs computer
--game_type defender #computer vs human
--game_type auto     #computer vs computer
# current default    #human vs human
```
