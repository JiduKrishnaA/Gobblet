# Gobblet

# ABSTRACT

The use of deep reinforcement learning techniques to master the game of Gobblet.
Gobblet is a strategic board game played on a 4x4 board with twelve pieces of varying sizes per
player. The objective is to form a row of four pieces, horizontally, vertically, or diagonally. To
propose and implement a deep reinforcement learning approach that models the game as a
Markov Decision Process (MDP), with states representing board positions, actions representing
piece placement or movement, and rewards indicating game outcomes. The method employs a
deep Q-network (DQN) to approximate the state-action value function and is trained using
experience replay and target networks for improved stability. Through analysis of the learned
policy, insights into the agent's strategy is inferred. It is observed that the agent learns to
prioritize protecting its own pieces while simultaneously obstructing its opponent's pieces. These
results highlight the effectiveness of deep reinforcement learning for achieving high-level
gameplay in Gobblet. This approach has the potential to be extended to other strategy board
games and can contribute to the development of advanced game-playing agents.

# METHODOLOGY
![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/e44d5035-dd3e-4d0c-be78-e2a6ce5fa782)

# images:

![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/bbc26da2-2de6-4246-ad3b-76052dd6ce2b)
Game Board

![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/9e92df3e-c0da-4b3b-902c-6d651a80d5d2)
Player Pieces

![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/77f921a1-991b-440c-a38a-5917596d2146)
Computer Pieces Preview of placing the piece

![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/f29a786e-a0d9-4083-b5b0-fae62440fdcd)
Piece Placement Preview Computer Wins the Game

![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/13d9e681-c918-4155-8a23-27ae2cb35e1e)
![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/a902b38b-94a6-4394-9d8a-07f6628a8411)
Computer winning state Player wins the game

![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/9c43bd7d-a741-4cef-bca9-d08349a40066)
![image](https://github.com/JiduKrishnaA/Gobblet/assets/101034086/203bdbdf-d432-469e-bf1f-e399398d856d)
Player winning state

# CONCLUSION
This project aimed to develop an AI agent capable of playing the 3x3 Gobblet board game and provide an enjoyable gaming experience for users. The project successfully achieved its objectives by implementing Tianshou DQN, a powerful deep reinforcement learning framework, to train the agent. The developed AI agent demonstrated remarkable learning capabilities and strategic decision-making during gameplay. The agent showcased adaptive behaviors, effectively utilizing its knowledge to make optimal moves and respond to changing game situations. The project has laid a solid foundation for further research and development, paving the way for advancements in AI-driven gaming and human-AI interactions. By combining the strengths of reinforcement learning algorithms and game-playing strategies, this project contributes to the broader field of artificial intelligence and opens up opportunities for further exploration in various domains, including gaming, decision-making, and multi-agent systems. Thus, this project has achieved its objectives, identified future directions for improvement and expansion, and holds great potential for further advancements in AI gaming. The knowledge gained from this project will serve as a valuable asset for future research and development endeavors in the field of artificial intelligence and game-playing agents.
