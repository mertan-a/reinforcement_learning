# Homework 2

### Algorithms:
* [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
  * [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
  * [Prioritized Replay buffer](https://arxiv.org/pdf/1511.05952.pdf)
  * [Duelling Networks](https://arxiv.org/pdf/1511.06581.pdf)
* [A3C](https://arxiv.org/pdf/1602.01783.pdf)

### Environments
* Bipedal Walker
* Lunar Lander
* Pong
* Breakout

> You may start the homework by reading the papers.
> Leave at least one week for atari training.

## Install
Create a virual environment

``` conda create -n blg604ehw2 python=3.7 ```

Activate it

``` source activate blg604ehw2 ```

Install the package at the current directory

``` pip install -e .```


<span style="color:red">**Note:**</span> If you get an error message while installing box2d:

__Unable to execute 'swig': No such file or directory__, try:

``` apt-get install swig ```


## What to upload?

- blg604ehw2 directory as zip
- Best agents for Pong and Breakout as another zip
- Ipython notebook
> For any question related with the homework, you can contact me tolgaokk@gmail.com