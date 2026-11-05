"""Core module."""

import json as js
import os
from typing import Optional

from . import utils
from .base import Base
from .utils import slugify as make_slug


def greet(name: str) -> str:
    """Return a greeting.

    Uses the slug of the name.
    """
    return format_message(make_slug(name))


def format_message(text):
    return f"Hello {text}"


class Greeter(Base):
    """Greets people."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or os.getenv("USER")

    @property
    def label(self):
        return self.name

    @label.setter
    def label(self, value):
        self.name = value

    def reset(self):
        self.name = None

    def run(self):
        self.save()
        self.reset()
        print(js.dumps({}), utils)
        return greet(self.label)

    @staticmethod
    def build(self):
        return self.run()


def shadowed(format_message):
    return format_message("x")


async def outer():
    def inner():
        return 1

    return inner()
