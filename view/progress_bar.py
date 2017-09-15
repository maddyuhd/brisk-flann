import progressbar


class progress:
    def __init__(self, msg, max):
        self.count = 0
        self.widgets = [str(msg) + ":", progressbar.Percentage(),
                        " ", progressbar.Bar(), " ", progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(maxval=max, widgets=self.widgets)
        self.pbar.start()

    def update(self):
        self.count += 1
        self.pbar.update(self.count)

    def finish(self):
        self.pbar.finish()
