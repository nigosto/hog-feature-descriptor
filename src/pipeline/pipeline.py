from .stage import Stage
from concurrent.futures import Future
from queue import Queue
from threading import Thread

class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, data):
        for stage in self.stages:
            data = stage.apply(data)

        return data

class ParallelPipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = [{"stage": stage, "queue": Queue(10)} for stage in stages]
        self.threads: list[Thread] = []

        self._start()

    def submit(self, data) -> Future:
        future = Future()
        self.stages[0]["queue"].put((data, future))

        return future

    def shutdown(self):
        self.stages[0]["queue"].put(None)
        for thread in self.threads:
            thread.join()

    def _start(self):
        for i, stage in enumerate(self.stages):
            input_queue = stage["queue"]
            output_queue = self.stages[i + 1]["queue"] if (i + 1) < len(self.stages) else None

            self.threads.append(Thread(
                target=self._process_stage,
                args=(stage["stage"], input_queue, output_queue),
                daemon=True
            ))
            self.threads[i].start()

    def _process_stage(self, stage: Stage, input_queue: Queue, output_queue: Queue):
        while True:
            item = input_queue.get()

            if item is None:
                if output_queue:
                    output_queue.put(None)
                input_queue.task_done()
                break

            image, future = item
            result = stage.apply(image)

            if output_queue:
                output_queue.put((result, future))
            else:
                future.set_result(result)

            input_queue.task_done()