import time


class Timer:
    def __init__(self):
        self.times = {}

    def start(self, name):
        self.times[name] = {"start": time.perf_counter(), "elapsed": None}

    def stop(self, name):
        if name in self.times and self.times[name]["start"] is not None:
            self.times[name]["elapsed"] = time.perf_counter() - self.times[name]["start"]

    def get(self, name, default=0.0):
        if name not in self.times or self.times[name]["elapsed"] is None:
            return default
        return self.times[name]["elapsed"]

    def summary(self):
        return {k: v["elapsed"] for k, v in self.times.items() if v["elapsed"] is not None}