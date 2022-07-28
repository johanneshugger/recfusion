from typing import Any, NamedTuple


class TimeStep(NamedTuple):
    done: Any
    reward: Any
    discount: Any
    observation: Any

    def first(self) -> bool:
        return self.done == False

    def mid(self) -> bool:
        return self.done == False

    def last(self) -> bool:
        return self.done == True


