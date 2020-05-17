import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class PG(tf.keras.Model):

    def __init__(self, state_size=4, action_size=2, layer_size=32):
        super(PG, self).__init__()

        # shared layers
        inputs = tf.keras.layers.Input(shape=[state_size,])
        x = tf.keras.layers.Dense(layer_size, activation="elu")(inputs)
        x = tf.keras.layers.Dense(layer_size, activation="elu")(x)

        # seperate heads
        actions_probs= tf.keras.layers.Dense(action_size, activation="softmax")(x)
        state_value = tf.keras.layers.Dense(1)(x)

        # we seperate them because we want to mess with gradients of the shared gradients;
        # a more efficient way is to use Model(inputs, outputs=[actions_probs, state_value])
        # and use 1 optimizer. Here we use seperate them to demostrate Duel Gradient Decent.
        self.policy_function=tf.keras.models.Model(inputs=inputs, outputs=actions_probs)
        self.value_function=tf.keras.models.Model(inputs=inputs, outputs=state_value)

    def call(self, state):
        '''Sample an action based on output probability'''
        action_probs = self.policy_function(state)
        action = tf.random.categorical(tf.math.log(action_probs), num_samples=1)
        action = tf.reshape(action, (1, 1))
        actions = tf.concat(values=[action, 1 - action], axis=1) # add the other action

        return actions


class Session():

    ''' Minus-state-value-bias version of REINFORCE

    The Monte Carlo returns(discounted sample rewards) are substracted by state value
    V(s) predicted from the value head.

    '''

    def __init__(self,
                 env,
                 discount_rate=0.95,
                 n_interations=50, # the number of iteration, each i runs n-episodes
                 n_episdoes_per_update=10, # number of episodes per update step
                 n_max_steps = 200, # maximum steps per episode
                 loss_functions=[tf.keras.losses.categorical_crossentropy,
                                 tf.keras.losses.mean_squared_error],
                 optimizers=[tf.keras.optimizers.Adam(), # for policy head
                             tf.keras.optimizers.Adam(), # for value head
                            ]):
        self.env=env
        self.discount_rate=discount_rate
        self.n_interations=n_interations
        self.n_episdoes_per_update=n_episdoes_per_update
        self.n_max_steps=n_max_steps
        self.loss_functions=loss_functions
        self.optimizers=optimizers

    def train(self, model, show_info=True):

        ''' use this to reproduce result

        env.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        '''

        mean_reward_log=[]

        for iteration in range(self.n_interations):

            # more efficient way would be to use placeholder array
            # this is to demonstrate the process
            all_episode_state_values, all_episode_rewards = [], []
            all_episode_gradients, all_episode_value_gradients = [], []
            for episode in range(self.n_episdoes_per_update):

                current_rewards, current_state_values = [], []
                current_gradients, current_value_gradients = [], []
                state_raw = self.env.reset()

                for step in range(self.n_max_steps):
                    state = tf.cast(state_raw[np.newaxis], dtype = 'float32')
                    with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
                        # compute policy loss, get its gradients
                        action_probs = model.policy_function(state) # compute to get 'action' and 'gradients' (1,1)
                        actions = model(state) # sample action (1, 1)
                        targets = 1 - tf.cast(actions, dtype='float32') # compute targets (1, 1)
                        policy_loss = self.loss_functions[0](targets, action_probs)
                        policy_gradients = policy_tape.gradient(policy_loss, model.policy_function.trainable_variables)

                        # observe next state and reward,
                        # compute value loss, get its gradientsr
                        state_value = model.value_function(state)
                        # observe next state, reward
                        action=int(np.argmax(actions, axis=1))
                        next_state, reward, done, info = self.env.step(action)
                        next_state_=tf.cast(next_state[np.newaxis], dtype = 'float32')
                        next_state_value = model.value_function(next_state_)
                        temporal_target=reward + self.discount_rate*next_state_value
                        value_loss= self.loss_functions[1](temporal_target, state_value)
                        value_gradients = value_tape.gradient(value_loss, model.value_function.trainable_variables)


                        # append experiences
                        current_state_values.append(float(state_value))
                        current_rewards.append(reward) # add current [...]
                        current_gradients.append(policy_gradients)
                        current_value_gradients.append(value_gradients)

                        state_raw=next_state
                        if done:
                            break

                all_episode_state_values.append(current_state_values)
                all_episode_rewards.append(current_rewards)
                all_episode_gradients.append(current_gradients)
                all_episode_value_gradients.append(current_value_gradients)

            # debugging
            if show_info:
                total_rewards = sum(map(sum, all_episode_rewards))
                mean_rewards = total_rewards / self.n_episdoes_per_update
                print("\rIteration: {}, mean rewards: {:.1f}".format(iteration, mean_rewards), end="")
                # make a record of it
                mean_reward_log.append(mean_rewards)

            # action score
            all_final_rewards = self._discount_biased_reward_over_n_episodes(all_episode_rewards, all_episode_state_values)
            # compute mean policy gradients
            all_mean_policy_grads = []
            for var_index in range(len(model.policy_function.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [final_reward * all_episode_gradients[episode_index][step][var_index]
                     for episode_index, final_rewards in enumerate(all_final_rewards)
                         for step, final_reward in enumerate(final_rewards)], axis=0)
                all_mean_policy_grads.append(mean_grads)

            # compute mean value gradients
            all_mean_value_grads = []
            for var_index in range(len(model.policy_function.trainable_variables)):
                mean_grads = tf.reduce_mean(
                    [all_episode_value_gradients[episode_index][step][var_index]
                     for episode_index, episode in enumerate(all_episode_gradients)
                        for step, var in enumerate(episode)], axis=0)

            # apply the double gradients to the model
            self.optimizers[0].apply_gradients(zip(all_mean_policy_grads, model.policy_function.trainable_variables))
            self.optimizers[1].apply_gradients(zip(all_mean_value_grads, model.value_function.trainable_variables))


        return mean_reward_log

    def _discount_biased_reward_over_n_episodes(self, all_episode_rewards, all_episode_state_values):
        # calculate discounted rewards for each episode
        all_discounted_episode_rewards = []
        for episode_rewards in all_episode_rewards:
            discounted_episode_rewards = np.empty(len(episode_rewards)) # initiate tensor
            cumulative_rewards = 0
            for i in reversed(range(len(episode_rewards))):
                cumulative_rewards = episode_rewards[i] + cumulative_rewards * self.discount_rate
                discounted_episode_rewards[i] = cumulative_rewards
            all_discounted_episode_rewards.append(discounted_episode_rewards)

        # have the discounted rewards substracted from state value V(s)
        biased_discounted_episode_rewards=[]
        for rewards, state_values in zip(all_discounted_episode_rewards, all_episode_state_values):
            biased_discounted_rewards=[]
            for reward, state_value in zip(rewards, state_values):
                biased_discounted_rewards.append(reward - state_value)
            biased_discounted_episode_rewards.append(biased_discounted_rewards)

        return biased_discounted_episode_rewards


if __name__ == '__main__':
    
    env = gym.make("CartPole-v1")

    ''' unlock this to reproduce result

    env.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    '''


    model_arg={
        "state_size":4,
        "action_size":2,
        "layer_size":16
        }

    session_arg={
        'env':env,
        'loss_functions':[tf.keras.losses.categorical_crossentropy, tf.keras.losses.mean_squared_error],
        'optimizers':[tf.keras.optimizers.Adam(lr=0.01), tf.keras.optimizers.Adam(lr=0.01)],
        'n_interations': 50,
        'discount_rate': 0.95,
        'n_episdoes_per_update': 10,
        'n_max_steps': 200,
        }

    pg=PG(**model_arg)
    sess=Session(**session_arg)
    mean_rewards=sess.train(pg)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards)
    plt.xlabel("Iteration(10 episodes each)", fontsize=14)
    plt.ylabel("Total Rewards", fontsize=14)
    plt.show()
