import requests

from pkg import greet
from pkg.core import Greeter


class App(Greeter):
    def main(self):
        return greet("world"), requests
