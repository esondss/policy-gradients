import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class REINFORCE(tf.keras.Model):
    
    def __init__(self, state_size=4, action_size=2, layer_size=16):
        super(REINFORCE, self).__init__()
        self.policy_function=tf.keras.models.Sequential([
            tf.keras.layers.Dense(layer_size, activation="elu",input_shape=(state_size,)),
            tf.keras.layers.Dense(layer_size, activation="elu"),
            tf.keras.layers.Dense(action_size, activation="softmax")])

    def call(self, state):
        action_probs=self.policy_function(state)
        # based on output probs, sample action.
        action = tf.random.categorical(tf.math.log(action_probs), num_samples=1)
        action = tf.reshape(action, (1, 1))
        actions = tf.concat(values=[action, 1 - action], axis=1)

        return actions

class Session():

    ''' Minus-bias version of REINFORCE

    The Monte Carlo returns(discounted sample rewards) are substracted by z-scores computed
    form a iteration of game plays.

    '''

    def __init__(self,
                 env,
                 discount_rate=0.95,
                 n_interations=50, # the number of iteration, each i runs n-episodes
                 n_episdoes_per_update=10, # number of episodes per update step
                 n_max_steps = 200, # maximum steps per episode
                 loss_function=tf.keras.losses.categorical_crossentropy,
                 optimizer=tf.keras.optimizers.Adam(lr=0.01)):
        self.env=env
        self.discount_rate=discount_rate
        self.n_interations=n_interations
        self.n_episdoes_per_update=n_episdoes_per_update
        self.n_max_steps=n_max_steps
        self.loss_function=loss_function
        self.optimizer=optimizer

    def train(self, policy_model, show_info=True):

        ''' use this to reproduce result

        env.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        '''

        mean_reward_log=[]

        for iteration in range(self.n_interations):

            all_rewards, all_gradients = [], []
            for episode in range(self.n_episdoes_per_update):

                current_rewards, current_gradients=[], []
                state_raw = self.env.reset()
                for step in range(self.n_max_steps):
                    state = tf.cast(state_raw[np.newaxis], dtype = 'float32')
                    with tf.GradientTape() as tape:

                        action_probs = policy_model.policy_function(state) # compute to get 'action' and 'gradients' (1,1)
                        actions =  policy_model(state) # sample action (1, 1)
                        targets = 1 - tf.cast(actions, dtype='float32') # compute targets (1, 1)

                        loss = self.loss_function(targets, action_probs)
                        gradients = tape.gradient(loss, policy_model.policy_function.trainable_variables)

                        #  revert one-hot to integer
                        action = int(np.argmax(actions, axis=1))

                        state_raw, reward, done, info = self.env.step(action)

                        current_rewards.append(reward) # add current [...]
                        current_gradients.append(gradients) # add current gradients [...]

                        if done:
                            break

                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

            # debugging
            if show_info:
                total_rewards = sum(map(sum, all_rewards))
                mean_rewards = total_rewards / self.n_episdoes_per_update
                print("\rIteration: {}, mean rewards: {:.1f}".format(iteration, mean_rewards), end="")
                # make a record of it
                mean_reward_log.append(mean_rewards)

            # action score
            all_final_rewards = self._discount_normalize_reward_over_n_episodes(all_rewards)

            # weight policy gradients by returns
            all_mean_grads = []
            for var_index in range(len(policy_model.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_gradients[episode_index][step][var_index]
                     for episode_index, final_rewards in enumerate(all_final_rewards)
                         for step, final_reward in enumerate(final_rewards)], axis=0)
                all_mean_grads.append(mean_grads)

            # apply the gradient to the model
            self.optimizer.apply_gradients(zip(all_mean_grads, policy_model.policy_function.trainable_variables))

        return mean_reward_log

    def _discount_normalize_reward_over_n_episodes(self, all_episode_rewards):
        # calculate discounted rewards for each episode
        all_discounted_episode_rewards = []
        for episode_rewards in all_episode_rewards:
            discounted_episode_rewards = np.empty(len(episode_rewards)) # initiate tensor
            cumulative_rewards = 0
            for i in reversed(range(len(episode_rewards))):
                cumulative_rewards = episode_rewards[i] + cumulative_rewards * self.discount_rate
                discounted_episode_rewards[i] = cumulative_rewards
            all_discounted_episode_rewards.append(discounted_episode_rewards)

        # calculate action mean & std rewards over n eposides
        flat_rewards = np.concatenate(all_discounted_episode_rewards)
        mean = np.mean(flat_rewards)
        std = np.std(flat_rewards)

        # normalizated score is the z-score
        normalized_discounted_episode_rewards=[]
        for rewards in all_discounted_episode_rewards:
            normalized_discounted_episode_rewards.append((rewards - mean) / (std))

        return normalized_discounted_episode_rewards


if __name__ == '__main__':
    
    
    env = gym.make("CartPole-v1")

    ''' unlock this to reproduce result

    env.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    '''

    model_arg={
        "state_size": 4,
        "action_size": 2,
        "layer_size": 16,
        }

    session_arg={
        'env':env,
        'loss_function':tf.keras.losses.categorical_crossentropy,
        'optimizer':tf.keras.optimizers.Adam(lr=0.01),
        'n_interations': 50,
        'discount_rate': 0.95,
        'n_episdoes_per_update': 10,
        'n_max_steps': 200,
    }

    policy_network=REINFORCE(**model_arg)
    sess=Session(**session_arg)

    rewards=sess.train(policy_network)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel("Iteration(10 episodes each)", fontsize=14)
    plt.ylabel("Rewards", fontsize=14)
    plt.show()
