import threading
from typing import Generic, TypeVar
from contextlib import contextmanager

T = TypeVar("T")

class ReadWriteLock(Generic[T]):
    def __init__(self, value: T):
        self._value = value
        self._read_count = 0
        self._write_count = 0
        self._waiting_writers = 0
        self._condition = threading.Condition()
        self._version = 0

    @property
    def value(self) -> T:
        with self._condition:
            return self._value

    @value.setter
    def value(self, new_value: T) -> None:
        with self._condition:
            self._value = new_value
            self._version += 1
            self._condition.notify_all()
    
    def set_silent(self, new_value: T) -> None:
        with self._condition:
            self._value = new_value

    def get_version(self) -> int:
        with self._condition:
            return self._version

    @contextmanager
    def read(self):
        with self._condition:
            while self._write_count != 0 or self._waiting_writers > 0:
                self._condition.wait()
            self._read_count += 1
        try:
            yield self._value
        finally:
            with self._condition:
                self._read_count -= 1
                if self._read_count == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self):
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._write_count != 0 or self._read_count > 0:
                    self._condition.wait()
                self._write_count += 1
            finally:
                self._waiting_writers -= 1
        try:
            yield self._value
        finally:
            with self._condition:
                self._write_count -= 1
                self._version += 1
                self._condition.notify_all()

    @contextmanager
    def wait_for_update(self, last_version: int):
        with self._condition:
            while self._version <= last_version:
                self._condition.wait()
            yield self._version