import datetime
import time
import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Prints and logs data within ReplayBuffer,
    parameters are modified directly by buffer threads
    """

    def __init__(self):
        self.datetime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        directory = 'logs'
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.writer = SummaryWriter(os.path.join(directory, self.datetime))

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
        Called asynchronously by ReplayBuffer to log data
        """
        elapsed_time = time.time() - self.start

        if self.loss != 0:
            self.writer.add_scalar('Loss', self.loss, self.total_updates)
            self.writer.add_scalar('Intrinsic_Loss', self.intr_loss, self.total_updates)
            self.writer.add_scalar('Reward', self.reward, self.total_updates)
            self.writer.add_scalar('Intrinsic_Reward', self.intrinsic, self.total_updates)
            self.writer.add_scalar('Epsilon', self.epsilon, self.total_updates)
            self.writer.add_scalar('Arm', self.arm, self.total_updates)
            self.writer.add_scalar('Replay_Ratio', self.replay_ratio, self.total_updates)

        # print('Elapsed: {:>8.4f} '
        #       'Updates: {:>8} '
        #       'Frames: {:>8} '
        #       'Loss: {:>10.8f} '
        #       'IntrLoss: {:>10.8f} '
        #       'Reward: {:>10.4f} '
        #       'Intrinsic: {:>10.4f} '
        #       'Epsilon: {:>8.3f} '
        #       'Arm: {:>8} '
        #       'ReplayRatio: {:>8.2f} '
        #       .format(elapsed_time,
        #               self.total_updates,
        #               self.total_frames,
        #               self.loss,
        #               self.intr_loss,
        #               self.reward,
        #               self.intrinsic,
        #               self.epsilon,
        #               self.arm,
        #               self.replay_ratio
        #               ),
        #       flush=True)

