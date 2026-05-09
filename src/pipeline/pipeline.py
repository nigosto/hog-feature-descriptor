from .stage import Stage


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, data):
        for stage in self.stages:
            data = stage.apply(data)

        return data
