# MEME (Efficient Memory-based Exploration agent)
An unofficial implementation of <a href="https://arxiv.org/pdf/2209.07550.pdf">MEME</a> (Efficient Memory-based Exploration agent) from DeepMind

## Learning Curves

<img src="https://github.com/YHL04/memeagent/blob/main/learning_curves/breakout_test_run.png" alt="drawing" width="600"/>

## TODO

- [X] Fix prioritized experience replay
- [X] Fix burnin functionality
- [X] Fix code for big burnin and rollout hyperparameter
- [ ] Find bugs and test for correctness

## Improvements

- [X] Bootstrapping with online network.
- [X] Target computation with tolerance.
- [X] Loss and priority normalization.
- [X] Cross-mixture training.
- [x] Normalizer-free torso network.
- [X] Shared torso with combined loss.
- [X] Robustifying behavior via policy distillation.

## Agent57 Original Code

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=YHL04&repo=agent57)](https://github.com/YHL04/agent57)

## Citations

```bibtex
@article{kapturowski2022human,
  title={Human-level Atari 200x faster},
  author={Kapturowski, Steven and Campos, V{\'\i}ctor and Jiang, Ray and Raki{\'c}evi{\'c}, Nemanja and van Hasselt, Hado and Blundell, Charles and Badia, Adri{\`a} Puigdom{\`e}nech},
  journal={arXiv preprint arXiv:2209.07550},
  year={2022}
}
```

