from __future__ import annotations
import argparse
import copy
from datetime import datetime
from datetime import timedelta
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests
import json
import os
import math
import csv
# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################


@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################


@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist, self.row+1+dist):
            for col in range(self.col-dist, self.col+1+dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1, self.col)
        yield Coord(self.row, self.col-1)
        yield Coord(self.row+1, self.col)
        yield Coord(self.row, self.col+1)

    def iter_adjacent_with_diagonal(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1, self.col)
        yield Coord(self.row, self.col-1)
        yield Coord(self.row+1, self.col)
        yield Coord(self.row, self.col+1)
        yield Coord(self.row+1, self.col+1)
        yield Coord(self.row+1, self.col-1)
        yield Coord(self.row-1, self.col+1)
        yield Coord(self.row-1, self.col-1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################


@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row+1):
            for col in range(self.src.col, self.dst.col+1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim-1, dim-1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################


class Heuristic(Enum):
    e0 = 0
    e1 = 1
    e2 = 2


@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None
    heuristic: Heuristic = Heuristic.e0

##############################################################################################################


@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0
    cumulative_evals: int = 0
    timer: datetime = None
##############################################################################################################


def game_heuristic_e0(game: Game) -> float:
    """
    returns the worth of the units of the attacker substracted by the worth of the units of the defender
    """
    unit_values: list[int] = [9999, 3, 3, 3, 3]

    attacker_eval = 0
    for (_, unit) in game.player_units(Player.Attacker):
        attacker_eval += unit_values[unit.type.value]

    defender_eval = 0
    for (_, unit) in game.player_units(Player.Defender):
        defender_eval += unit_values[unit.type.value]

    return attacker_eval - defender_eval


def game_heuristic_e1(game: Game) -> float:
    """
    This heuristic evaluates:
     1. the number of ennemy units near the respective player's AI
     2. If a Virus unit is close to the defender AI, adding a large positive number to the heuristic.
     3. If a Virus unit is close to Tech unit, a negative number is added
     4. If a Virus unit is close to Program unit, a positive number is added
    """

    ennemies_near_AI = 0
    virus_att_AI = 0
    virus_near_Tech = 0
    virus_near_program = 0
    for (coords, unit) in game.player_units(Player.Attacker):
        if unit.type.value == 0:
            in_near_range_of_AI = list([x for x in coords.iter_range(2)])
            for i in in_near_range_of_AI:
                if game.get(i) is not None:
                    if game.get(i).player.name != "Attacker":
                        ennemies_near_AI += 1
        if unit.type.value == 2:
            possible_1_shot = list(
                [x for x in coords.iter_adjacent_with_diagonal()])
            for i in possible_1_shot:
                if game.get(i) is not None:
                    if game.get(i).player.name != "Attacker" and game.get(i).type.value == 0:
                        virus_att_AI = 1000
                    if game.get(i).player.name != "Attacker" and game.get(i).type.value == 1:
                        virus_near_Tech = -50
                    if game.get(i).player.name != "Attacker" and game.get(i).type.value == 3:
                        virus_near_program += 25

    ennemies_near_def_AI = 0
    ennemies_long_range_def_AI = 0
    for (coords, unit) in game.player_units(Player.Defender):
        if unit.type.value == 0:
            in_range_of_AI = list([x for x in coords.iter_range(2)])
            for i in in_range_of_AI:
                if game.get(i) is not None:
                    if game.get(i).player.name != "Defender":
                        ennemies_near_def_AI += 1
            in_long_range_of_AI = list([x for x in coords.iter_range(3)])
            for i in in_long_range_of_AI:
                if game.get(i) is not None:
                    if game.get(i).player.name != "Defender":
                        ennemies_long_range_def_AI += 1
    e2_score = game_heuristic_e2(game)

    return e2_score + 5*(ennemies_near_def_AI - ennemies_near_AI) + ennemies_long_range_def_AI + virus_att_AI + virus_near_program + virus_near_Tech


def game_heuristic_e2(game: Game) -> float:
    """
    Evaluating score based on Attacker's vs Defender's units total health pool.
    Result with higher value is best for Attacker, Lower value for defender.
    Modifiers are used to influence decisions.
    """

    ai_mod = 1000  # Prioritize AI's safety.
    # Low value to avoid attacking Firewalls if possible or use it more by defender.
    firewall_mod = 1
    program_mod = 3  # Average unit but capable even against Virus/Tech
    other_mod = 5  # For Virus and Tech
    attacker_health = 0
    # Calculate the sum of health attacker unit's has at measured node.
    for (_, unit) in game.player_units(Player.Attacker):
        # Get AI's health
        if str(unit.type) == "UnitType.AI":
            attacker_health += unit.health * ai_mod
        # Get Firewall's health
        elif str(unit.type) == "UnitType.Firewall":
            attacker_health += unit.health * firewall_mod
        # Get Program's health
        elif str(unit.type) == "UnitType.Program":
            attacker_health += unit.health * program_mod
        # Get Virus and Tech's health
        else:
            attacker_health += unit.health * other_mod

    defender_health = 0
    # Calculate the sum of health defender unit's has at measured node.
    for (_, unit) in game.player_units(Player.Defender):
        # Get AI's health
        if str(unit.type) == "UnitType.AI":
            defender_health += unit.health * ai_mod
        # Get Firewall's health
        elif str(unit.type) == "UnitType.Firewall":
            defender_health += unit.health * firewall_mod
        # Get Program's health
        elif str(unit.type) == "UnitType.Program":
            defender_health += unit.health * program_mod
        # Get Program, Virus and Tech's health
        else:
            defender_health += unit.health * other_mod

    # Returning evaluation score of node
    return float(attacker_health-defender_health)


@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    moves_history = []
    board_history = []
    heuristic_history = []
    time_history = []

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(
            player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(
            player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(
            player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md-1, md),
                 Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md-1),
                 Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md-2, md),
                 Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md-2),
                 Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md-1, md-1),
                 Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair. TODO: 
        1. The destination must be free (no other unit is on it).
        2. Units are said to be engaged in combat if an adversarial unit is adjacent (in any of the 4 directions). 
            If an AI, a Firewall or a Program is engaged in combat, they cannot move.
            The Virus and the Tech can move even if engaged in combat.
        3. The attacker’s AI, Firewall and Program can only move up or left.
            The Tech and Virus can move left, top, right, bottom.
        4. The defender’s AI, Firewall and Program can only move down or right.
            The Tech and Virus can move left, top, right, bottom.
        """
        # Validates whether the coordinates are inside the board.
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        # Fetches unit at source coordinates.
        unit = self.get(coords.src)
        # Validates that there is a unit at the source coordinates and if it belongs to the player.
        if unit is None or unit.player != self.next_player:
            return False

        # Check if player intends to self destruct
        if coords.dst == coords.src:
            return True
        # Get player type
        player_type = unit.player
        # Get unit type
        unit_type = unit.type
        # Fetches unit at destination coordinates.
        unit_destination = self.get(coords.dst)

        # Check if unit is engaged in combat
        adjacent_coords = list([x for x in coords.src.iter_adjacent()])

        # Verify if dest coords are adjacent
        if coords.dst not in adjacent_coords and coords.dst != coords.src:
            return False

        # attack or repairing a unit on an adjacent field is always a valid move
        if unit_destination != None:
            if unit_destination.player == self.next_player:
                return bool(self.is_repairable(coords))
            else:
                return True

        # Verify that movement is in the right direction
        if (unit_type != UnitType.Tech and unit_type != UnitType.Virus):

            # check if unit is enaged in combat
            for coord in adjacent_coords:
                unit = self.get(coord)
                if unit == None:
                    continue
                if unit.player != self.next_player:
                    return False

            if (player_type.value == 0):  # Is attacker
                if coords.src.row < coords.dst.row or coords.src.col < coords.dst.col:  # Unit is moving down -> illegal
                    return False
            else:  # Is defender
                if coords.src.row > coords.dst.row or coords.src.col > coords.dst.col:
                    return False
        return True

    def perform_self_destruct(self, coord: Coord):
        """Makes a unit on a given field destroy itself and damages all units on diagonal and adjacant fields."""
        self.mod_health(coord, -9)
        for adjacent_coord in coord.iter_adjacent_with_diagonal():
            adjacent_unit = self.get(adjacent_coord)
            if adjacent_unit != None:
                self.mod_health(adjacent_coord, -2)

    def perform_move(self, coords: CoordPair, record_move=True) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair."""
        if self.is_valid_move(coords):
            dst_unit = self.get(coords.dst)

            # if action = move
            if dst_unit == None:
                if record_move:
                    self.record_move(coords, action="move")
                self.set(coords.dst, self.get(coords.src))
                self.set(coords.src, None)
                self.record_board()
                return (True, "")

            # if action = attack
            if dst_unit.player != self.next_player:
                if record_move:
                    self.record_move(coords, action="attack")
                self.attack(coords)
                self.record_board()
                return (True, "")  # TODO check if return is correct

            # checks if unit on destination belongs to player
            if dst_unit.player == self.next_player:

                # if action = self_destruct
                if coords.src == coords.dst:
                    if record_move:
                        self.record_move(coords, action="self-destruct")
                    self.perform_self_destruct(coords.src)
                    self.record_board()
                    return (True, "")  # TODO check if return is correct

                # if action = repair
                else:
                    restored_health = self.is_repairable(coords)
                    if restored_health:
                        self.mod_health(coords.dst, restored_health)
                        if record_move:
                            self.record_move(coords, action="repair")
                        self.record_board()
                        return (True, "")  # TODO check if return is correct
                    else:
                        return (False, "invalid move")

            # if action = self_destruct
            if coords.src == coords.dst:
                if record_move:
                    self.record_move(coords, action="self-destruct")
                self.perform_self_destruct(coords.src)
                self.record_board()

            raise AssertionError("A valid move should always be handled.")
        return (False, "invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success, result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ", end='')
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta."""
        start_time = datetime.now()
        self.stats.timer = datetime.now()
        # (score, move, avg_depth) = self.random_move()
        (score, move, avg_depth) = self.alphabeta_move()
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        self.heuristic_history.append(score)
        # Prints the number of evaluations made per depth
        print(f"Evals per depth: ", end='')
        c = 1

        for k in sorted(self.stats.evaluations_per_depth.keys(), reverse=True):
            print(f"{c}:{self.stats.evaluations_per_depth[k]} ", end='')
            c += 1
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        # Calculate average branching factor:
        avg_branching = 0
        prev_k = None
        for k in sorted(self.stats.evaluations_per_depth.items(), reverse=True):
            if prev_k == None:
                depth, prev_k = k
            else:
                # Old number of nodes stored in temp
                prev_prev_k = prev_k
                # New number of ndoes stored in prev_k
                depth, prev_k = k
                avg_branching += (prev_k/prev_prev_k)
        if len(self.stats.evaluations_per_depth.keys()) == 0:
            print(f"Average branching factor: {avg_branching:0.1f}")
        else:
            print(
                f"Average branching factor: {avg_branching / len(self.stats.evaluations_per_depth.keys()):0.1f}")
        # Clear evaluations_per_depth stats after every turn
        self.stats.evaluations_per_depth.clear()
        # Reset timer for next alphabeta call:
        self.stats.timer = None
        # Add total number of evaluations of that turn to the total number of evaluations during the game
        self.stats.cumulative_evals += total_evals
        print(f"Cumulative evals: {self.stats.cumulative_evals}")

        # Prints number of evaluations performed
        if self.stats.total_seconds > 0:
            print(
                f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(
                    f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

    def attack(self, coords: CoordPair):
        """lets a unit on one coordinate attack a unit on another. Requires pre-check if attack is possible."""
        attacking_unit = self.get(coords.src)
        defending_unit = self.get(coords.dst)
        attack_damage = attacking_unit.damage_amount(defending_unit)
        self_damage = defending_unit.damage_amount(attacking_unit)
        self.mod_health(coords.dst, -attack_damage)
        self.mod_health(coords.src, -self_damage)

    def is_repairable(self, coords: CoordPair) -> int:
        """lets a unit on one coordinate repair a unit on another. Requires pre-check if repair is possible. 
        Returns if repair was successful."""
        repairing_unit = self.get(coords.src)
        target_unit = self.get(coords.dst)
        restored_health = repairing_unit.repair_amount(target_unit)
        # print(restored_health, "restored health")
        return restored_health
        if restored_health == 0:
            return False  # if the health change is 0, it is not a valid move
        self.mod_health(coords.dst, restored_health)
        return True

    def record_move(self, coords: CoordPair, action: str):
        unit = self.get(coords.src)
        player = unit.player.name
        if action == "move":
            stringAction = "moved from " + \
                str(coords.src) + " to " + str(coords.dst)
        elif action == "attack":
            stringAction = str(coords.src) + " attacked " + str(coords.dst)
        elif action == "repair":
            stringAction = str(coords.src) + " repaired " + str(coords.dst)
        elif action == "self-destruct":
            stringAction = str(unit) + " at " + \
                str(coords.src) + " self-destructed!"

        stringAction = "\nTurn #" + str(self.turns_played) + \
            " " + player + " " + stringAction + "\n\n"

        self.moves_history.append(stringAction)

        return

    def record_board(self):
        self.board_history.append(self.to_string())

    def print_history(self):
        for i in self.moves_history:
            print(i)

    def to_eval_data(self):
        """
        Appends information to two files 
        (1) heuristic_data.txt
        (2) performance_data.txt
        """

        try:
            winner = self.has_winner()
            best_defender_val = min(self.heuristic_history)
            best_attacker_val = max(self.heuristic_history)
            p_text = [self.has_winner(), len(self.moves_history),
                      best_defender_val,  best_attacker_val]
            with open('performance_data.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(p_text)
                f.close()
        except Exception as e:
            print('Could not output performance data to file.', e)
        try:
            with open('heuristic_data.csv', 'a+') as f:
                h_text = self.heuristic_history[1::2]
                writer = csv.writer(f)
                writer.writerow(h_text)
                f.close()
        except Exception as e:
            print('Could not output game trace to file.', e)

    def game_trace_to_file(self, initialBoard):
        try:
            # Build the variable with all game options + initial configuration

            game_timeout = self.options.max_time
            num_of_turns = self.options.max_turns
            game_nr = random.randint(1, 1000)
            FOLDERNAME = "gametrace"
            filename = f"gameTrace{game_nr}-" + \
                str(self.options.alpha_beta).lower() + "-" + \
                str(game_timeout)+"-"+str(num_of_turns)+".txt"
            path = os.path.join(FOLDERNAME, filename)
            f = open(path, "w")

            text = "Game Trace" + "\n" + \
                "---------------------------------------------------------" + "\n"
            text += "The game parameters are: \n\n"
            text += "Maximum time allowed: " + \
                str(self.options.max_time) + \
                "    *Note this does not apply to Human players.\n"
            text += "Maximum turns allowed: " + \
                str(self.options.max_turns) + "\n"

            if (str(self.options.game_type) == 'GameType.AttackerVsDefender'):  # Human game
                text += "Game mode: player 1 = Human & player 2 = Human"
            else:
                text += "Alpha-beta is not yet implemented, AI players do not yet use heuristics - they only perform randomly chosen moves.\n" + \
                    "*For these reasons, we do not track evals by depth, etc.\n\n"
                """
                if (self.options.alpha_beta):
                    text += "Alpha-beta is on"
                else:
                    text += "Alpha-beta is off"
                """
                if (str(self.options.game_type) == 'GameType.AttackerVsComp'):
                    text += "Game mode: player 1 = Human & player 2 = AI"
                elif (str(self.options.game_type) == 'GameType.CompVsDefender'):
                    text += "Game mode: player 1 = AI & player 2 = Human"
                else:
                    text += "Game mode: player 1 = AI & player 2 = AI"
            text += "\n---------------------------------------------------------" + "\n"
            text += "Initial Board:" + "\n"
            text += str(initialBoard) + "\n"
            text += "---------------------------------------------------------" + "\n"
            text += "Every turn information:"
            # text += self.to_string()

            for i in range(0, len(self.moves_history)):
                text += "\n\n"
                text += str(self.moves_history[i])
                text += self.board_history[i]
            winner = self.has_winner()
            text += "\n\n" + winner.name + " won the game in " + \
                str(self.turns_played - 1) + " turns."
            f.write(text)
            f.close()
        except Exception as e:
            print('Could not output game trace to file.', e)

    def initial_Board(self):
        return self.to_string()

    def calc_heuristic(self) -> float:
        if self.options.heuristic == 0:
            return game_heuristic_e0(self)
        if self.options.heuristic == 1:
            return game_heuristic_e1(self)
        if self.options.heuristic == 2:
            return game_heuristic_e2(self)
        raise AssertionError("There should always be a selected heuristic.")

    def alphabeta_move(self, node: Game = None, depth: int = None, alpha: float = None, beta: float = None, player: Player = None, pruning: bool = None) -> Tuple[int, CoordPair | None, float]:
        """
        """
        # TODO think about min depth
        if node == None:
            node = self
        if depth == None:
            depth = self.options.max_depth
        if alpha == None:
            alpha = -math.inf
        if beta == None:
            beta = math.inf
        if player == None:
            player = self.next_player
        if pruning == None:
            pruning = True

        # Call function to generate list of candidate moves (child nodes).
        move_candidates = [
            move_candidate for move_candidate in node.move_candidates()]

        # Check if its leaf nodes and evaluate heuristics if so.
        if depth == 0 or node.has_winner() or datetime.now() - self.stats.timer > timedelta(seconds=self.options.max_time):  # TODO or if node is terminal
            return (node.calc_heuristic(), None, 0)

        if player == Player.Attacker:
            v = -math.inf
            performed_move = None
            for possible_move in move_candidates:
                child_node = node.clone()
                child_node.perform_move(possible_move, record_move=False)
                child_node.next_turn()

                # Recusive call to the next depth of current node (parent).
                (child_node_eval, suggested_move, average_depth) = self.alphabeta_move(
                    child_node, depth - 1, alpha, beta, Player.Defender)

                if child_node_eval > v:
                    v = child_node_eval
                    performed_move = possible_move
                    # v = max(v, child_node_eval)
                alpha = max(alpha, v)
                if pruning:
                    if beta <= alpha:
                        break
            if depth in self.stats.evaluations_per_depth:
                self.stats.evaluations_per_depth[depth] += len(move_candidates)
            else:
                self.stats.evaluations_per_depth[depth] = len(move_candidates)
            return (v, performed_move, 0)  # TODO handle average depth
        else:
            v = math.inf
            performed_move = None
            for possible_move in move_candidates:
                child_node = node.clone()
                child_node.perform_move(possible_move, record_move=False)
                child_node.next_turn()

                (child_node_eval, suggested_move, average_depth) = self.alphabeta_move(
                    child_node, depth - 1, alpha, beta, Player.Attacker)

                if child_node_eval < v:
                    v = child_node_eval
                    performed_move = possible_move
                    # v = min(v, self.alphabeta(child_node, depth -1, alpha, beta, Player.Attacker))
                beta = min(beta, v)
                if pruning:
                    if beta <= alpha:
                        break
            # Accumulates number of possible moves at every depth for all branches
            if depth in self.stats.evaluations_per_depth:
                self.stats.evaluations_per_depth[depth] += len(move_candidates)
            else:
                self.stats.evaluations_per_depth[depth] = len(move_candidates)
            return (v, performed_move, 0)  # TODO handle average depth

##############################################################################################################


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--alpha_beta', type=bool, default=True,
                        help='determines if alpha-beta pruning is turned on or off')
    parser.add_argument('--heuristic', type=int, default=0,
                        help='heuristic applied: 0 = e0, 1 = e1, 2 = e2')
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual",
                        help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--max_turns', type=int, default=100,
                        help='max number of turns: ex: 100')

    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if args.max_turns is not None:
        options.max_turns = args.max_turns
    if args.alpha_beta is not None:
        options.alpha_beta = args.alpha_beta
    if args.heuristic is not None:
        options.heuristic = args.heuristic

    # create a new game
    game = Game(options=options)
    print()
    print()
    initialBoard = game.initial_Board()
    game.turns_played += 1
    # the main game loop
    while True:
        print()
        print(game)
        print(len([move for move in game.move_candidates()]))
        for move in game.move_candidates():
            # print(move)
            pass
        print(game.suggest_move())

        winner = game.has_winner()
        if winner is not None:
            # Output moves to file
            game.print_history()
            game.game_trace_to_file(initialBoard)
            game.to_eval_data()

            print(f"{winner.name} wins!")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

##############################################################################################################


if __name__ == '__main__':
    for i in range(0, 20):
        main()
