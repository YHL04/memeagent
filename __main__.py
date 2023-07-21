

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import portpicker
import os
import time

from agent import Learner


def run_worker(
    rank,
    env_name,
    num_actors,
    buffer_size,
    batch_size,
    burnin,
    rollout
):
    """
    Workers:
        - Learner.run()
        - Actor.run()

    Threads:
        - Learner.answer_requests(time.sleep(0.0001))
        - Learner.prepare_data(time.sleep(0.001))
        - ReplayBuffer.add_data(time.sleep(0.001))
        - ReplayBuffer.prepare_data(time.sleep(0.001))
        - ReplayBuffer.update_data(time.sleep(0.001))
        - ReplayBuffer.log_data(time.sleep(10))

    """

    if rank == 0:
        # create Learner in a remote location
        rpc.init_rpc("learner", rank=rank, world_size=1+num_actors)

        learner_rref = rpc.remote(
            "learner",
            Learner,
            args=(env_name,
                  num_actors,
                  buffer_size,
                  batch_size,
                  burnin,
                  rollout
                  ),
            timeout=0
        )
        learner_rref.remote().run()

        # Start training loop
        while True:
            time.sleep(1)

    else:
        # Create actor in a remote location
        rpc.init_rpc(f"actor{rank-1}", rank=rank, world_size=1+num_actors)

    rpc.shutdown()


def main(env_name,
         num_actors,
         buffer_size,
         batch_size,
         burnin,
         rollout,
         ):
    # set localhost and port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(portpicker.pick_unused_port())

    mp.spawn(
        run_worker,
        args=(env_name,
              num_actors,
              buffer_size,
              batch_size,
              burnin,
              rollout
              ),
        nprocs=1+num_actors,
        join=True
    )


if __name__ == "__main__":
    main(env_name="BreakoutDeterministic-v4",
         num_actors=2,
         buffer_size=400_000,
         batch_size=64,
         burnin=0,
         rollout=10
         )
