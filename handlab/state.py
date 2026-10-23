from enum import Enum


class State(str, Enum):
    NORMAL = 'NORMAL'
    SUSPECT = 'SUSPECT'
    LOST = 'LOST'
    RECOVERING = 'RECOVERING'


class StateMachine:
    def __init__(self, confirmation=3):
        self.state = State.LOST
        self.clean_frames = 0
        self.confirmation = confirmation

    def transition(self, state):
        previous = self.state
        self.state = state
        if state != State.SUSPECT:
            self.clean_frames = 0
        return None if previous == state else f'{previous.value}->{state.value}'

    def normal(self):
        if self.state == State.SUSPECT:
            self.clean_frames += 1
            if self.clean_frames < self.confirmation:
                return None
        return self.transition(State.NORMAL)

    def suspect(self):
        self.clean_frames = 0
        return self.transition(State.SUSPECT)
