

from .bandit import UCB
from .runningmeanstd import RunningMeanStd
from .soft_watkins import compute_retrace_loss
from .value_rescale import rescale, inv_rescale
from .schedules import get_betas, get_discounts
from .misc import tonumpy, tosqueeze, tounsqueeze, totensor, toconcat

