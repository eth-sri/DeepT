class FakeLogger:
    def __init__(self):
        pass

    def _clear(self):
        pass

    def get_summary_sum(self, s, length):
        return 0

    def next_epoch(self):
        pass

    def next_step(self, out):
        pass

    def add_valid(self, out):
        pass

    def add_test(self, out):
        pass

    def save_valid(self, log=False):
        pass

    def save_test(self, log=False):
        pass

    def get_epoch(self):
        return 1

    def write(self, *all_text):
        print(*all_text)