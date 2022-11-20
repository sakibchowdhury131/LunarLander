import gym
from Brain import Agent
from utils import plotLearning
import numpy as np



if __name__ == '__main__':
    ENV_NAME = 'LunarLander-v2'
    LOAD_TRAINED_MODEL = True
    MODEL_PATH = ENV_NAME + '.pkl'
    env = gym.make(ENV_NAME)
    #agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.sample().shape[0], eps_end=0.35,
    #              input_dims=[env.observation_space.sample().shape[0]], lr=0.001)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.2,
                  input_dims=[8], lr=0.001, eps_dec=5e-4)
    scores, eps_history = [], []
    n_games = 2500

    if LOAD_TRAINED_MODEL:
        agent.load_experience(MODEL_PATH)
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            env.render()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        if i%10 == 1:
            agent.save_experience(PATH=MODEL_PATH)
    x = [i+1 for i in range(n_games)]
    filename = ENV_NAME + '.png'
    plotLearning(x, scores, eps_history, filename)