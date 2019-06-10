import math
import numpy as np
import pandas as pd
from IPython.display import display


def html_table(lol):
    html = '<table>\n'
    for sublist in lol:
        html += '  <tr><td>'
        html += '    </td><td>'.join(sublist)
        html += '  </td></tr>\n'
    html += '</table>'
    return html


class GridWorld:
    def __init__(self, initial_values=0.):
        self.states = [['A', 'C', 'Gold'], ['B', 'D', 'Pit']]
        self.dim = len(self.states), len(self.states[0])
        self.actions = {'North': np.array((-1, 0)), 'West': np.array((0, -1)), 'East': np.array((0, 1)),
                        'South': np.array((1, 0))}
        self.rewards = {'Gold': 20, 'Pit': -10, 'Move': 0}
        self.transitions = {'straight': 0.8, 'left': 0.1, 'right': 0.1}
        self.discount = 1

        self.rotations = {'straight': np.identity(2, dtype=int), 'back': -np.identity(2, dtype=int),
                          'left': np.array([[0, -1], [1, 0]], dtype=int),
                          'right': np.array([[0, 1], [-1, 0]], dtype=int)}

        if initial_values:
            self.value_functions = [initial_values, initial_values]
        else:
            self.value_functions = [[[0 for v in row] for row in self.states],
                                    [[0 for v in row] for row in self.states]]

        self.residuals = [[math.inf for v in row] for row in self.states]

    def __str__(self):
        return str(self.states)

    def _repr_html_(self):
        strings = [['{:.3f}'.format(v) for v in row] for row in self.value_functions[0]]
        return html_table(strings)

    def get_state_pos(self, state):
        for r in range(len(self.states)):
            for c in range(len(self.states[r])):
                if self.states[r][c] == state:
                    return np.array((r, c))

    def get_pos_state(self, pos):
        return self.states[pos[0]][pos[1]]

    def valid_position(self, pos):
        return 0 <= pos[0] < self.dim[0] and 0 <= pos[1] < self.dim[1]

    def get_transitions(self, state, action):
        outcomes = []

        if state in ['Pit', 'Gold']:
            return [(1, state)]

        current_pos = self.get_state_pos(state)
        act_dir = self.actions[action]

        for rot_name, prob in self.transitions.items():
            new_pos = current_pos + self.rotations[rot_name].dot(act_dir)
            if self.valid_position(new_pos):
                outcomes.append((prob, self.get_pos_state(new_pos)))
            else:
                outcomes.append((prob, state))
        return outcomes

    def get_reward(self, state, next_state):
        if state in ['Gold', 'Pit']:
            return 0

        if next_state in ['Gold', 'Pit']:
            return self.rewards[next_state]
        else:
            return self.rewards['Move']

    def get_state_value(self, state, vf_id=0):
        pos = self.get_state_pos(state)
        return self.value_functions[vf_id][pos[0]][pos[1]]

    def set_value(self, state, value, vf_id=0):
        pos = self.get_state_pos(state)
        self.value_functions[vf_id][pos[0]][pos[1]] = value

    def backup(self, state):
        new_value = -math.inf
        best_action = None
        for action in self.actions:
            action_value = 0
            for prob, next_state in self.get_transitions(state, action):
                action_value += prob * (
                        self.get_reward(state, next_state) + self.discount * self.get_state_value(next_state))

            if action_value > new_value:
                new_value = action_value
                best_action = action
        return new_value, best_action

    def async_VI(self, order):
        for state in order:
            new_value, best_action = self.backup(state)
            self.set_value(state, new_value)


if __name__ == '__main__':
    gw = gridworld()
    gw.async_VI(order='ABCDABCD')
    display(gw)
    gw.backup('D')
