import argparse
import sys
from typing import Tuple

import torch
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, MultiAgentPolicyManager

import gobblet
from manual_policy import ManualGobbletPolicy  # noqa 401
from collector_manual_policy import ManualPolicyCollector
from greedy_policy_tianshou import GreedyPolicy


import tkinter as tk

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument(
        "--lr", type=float, default=1e-4
    )  # TODO: Changing this to 1e-5 for some reason makes it pause after 3 or 4 epochs
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["human", "rgb_array", "text", "text_full"],
        help="Choose the rendering mode for the game.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to enable to print extra debugging info",
    )
    parser.add_argument(
        "--self_play",
        action="store_true",
        help="Flag to enable training via self-play (as opposed to fixed opponent)",
    )
    parser.add_argument(
        "--cpu-players",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of CPU players (options: 1, 2)",
    )
    parser.add_argument(
        "--player",
        type=int,
        default=0,
        choices=[0, 1],
        help="Choose which player to play as: red = 0, yellow = 1",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Flag to save a recording of the game (game.gif)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Search depth for greedy agent. Options: 1,2,3",
    )
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents() -> Tuple[BasePolicy, list]:
    env = get_env()
    agents = [GreedyPolicy(depth=args.depth), GreedyPolicy(depth=args.depth)]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, env.agents


def get_env(render_mode=None, args=None):
    return PettingZooEnv(gobblet.env(render_mode=render_mode, args=args))


def show_popup(message):
    root = tk.Tk()
    root.title("Game Result")
    
    label = tk.Label(root, text=message, padx=20, pady=10)
    label.pack()
    
    ok_button = tk.Button(root, text="OK", command=root.destroy)
    ok_button.pack(pady=10)
    
    root.mainloop()

def play() -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode=args.render_mode, args=args)])

    policy, agents = get_agents()
    collector = ManualPolicyCollector(
        policy, env, exploration_noise=True
    )  # Collector for CPU actions

    pettingzoo_env = env.workers[
        0
    ].env.env  # DummyVectorEnv -> Tianshou PettingZoo Wrapper -> PettingZoo Env

    manual_policy = ManualGobbletPolicy(
        env=pettingzoo_env, agent_id=args.player
    )  # Gobblet keyboard input requires access to raw_env (uses functions from board)

    while pettingzoo_env.agents:
        agent_id = collector.data.obs.agent_id
        # If it is the players turn and there are less than 2 CPU players (at least one human player)
        if agent_id == pettingzoo_env.agents[args.player]:
            observation = {
                "observation": collector.data.obs.obs.flatten(),
                "action_mask": collector.data.obs.mask.flatten(),
            }  # PettingZoo expects a dict with this format
            action = manual_policy(observation, agent_id)

            result = collector.collect_result(
                action=action.reshape(1), render=args.render
            )
        else:
            result = collector.collect(n_step=1, render=args.render)


        if collector.data.terminated or collector.data.truncated:
            rews, lens = result["rews"], result["lens"]
            print(
                f"\nFinal reward: {rews[:, args.player].mean()}, length: {lens.mean()} [Human]"
            )
            print(
                f"Final reward: {rews[:, 1-args.player].mean()}, length: {lens.mean()} [{type(policy.policies[agents[1-args.player]]).__name__}]"
            )



            # Show popup based on the result
            if rews[:, args.player].mean() == 1:
                show_popup("Human wins")
            elif rews[:, 1-args.player].mean() == 1:
                show_popup("Computer wins")
            sys.exit()

if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()
    print("Starting game...")
    play()