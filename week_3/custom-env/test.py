import custom_project_env
from agent import QLearningAgent, PotentialBasedAgent
import gymnasium as gym


env = gym.make("custom_project_env/GridWorld-v0", render_mode="human", size=10)

# agent = QLearningAgent(env)
agent2 = PotentialBasedAgent(env)

# agent.train(10000)
# average_reward, success_rate, average_steps = agent.evaluate(episodes=100)
# print(f"Average Reward: {average_reward}")
# print(f"Success Rate: {success_rate * 100}%")
# print(f"Average steps: {average_steps}")

print("Agent2 evaluation")
average_reward, success_rate, average_steps = agent2.evaluate(episodes=10)
print(f"Average Reward: {average_reward}")
print(f"Success Rate: {success_rate * 100}%")
print(f"Average steps: {average_steps}")

env.close()
