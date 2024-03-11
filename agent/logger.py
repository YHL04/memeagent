

import datetime
import time


class Logger:
    """
    Prints and logs data withing ReplayBuffer,
    parameters are modified directly by buffer threads
    """

    def __init__(self):
        self.datetime = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
        self.file = open(f"logs/{self.datetime}", "w")

        self.total_frames = 0
        self.total_updates = 0
        self.loss = 0
        self.intr_loss = 0
        self.reward = 0
        self.intrinsic = 0
        self.epsilon = 0
        self.arm = 0
        self.replay_ratio = 0

        self.start = time.time()

    def print(self):
        """
        Called asynchronously by ReplayBuffer to prints logs and store data in logs directory
        """
        elapsed_time = time.time() - self.start

        if self.loss != 0:
            self.file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(elapsed_time,
                                                                      self.total_updates,
                                                                      self.total_frames,
                                                                      self.loss,
                                                                      self.intr_loss,
                                                                      self.reward,
                                                                      self.intrinsic,
                                                                      self.arm))
            self.file.flush()

        print('Elapsed: {:>8.4f} '
              'Updates: {:>8} '
              'Frames: {:>8} '
              'Loss: {:>10.8f} '
              'IntrLoss: {:>10.8f} '
              'Reward: {:>10.4f} '
              'Intrinsic: {:>10.4f} '
              'Epsilon: {:>8.3f} '
              'Arm: {:>8} '
              'ReplayRatio: {:>8.2f} '
              .format(elapsed_time,
                      self.total_updates,
                      self.total_frames,
                      self.loss,
                      self.intr_loss,
                      self.reward,
                      self.intrinsic,
                      self.epsilon,
                      self.arm,
                      self.replay_ratio
                      ),
              flush=True)

