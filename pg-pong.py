""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def process_frame(frame, previously_processed_frame):
    # crop
    processed_frame = frame[35:195]
    # reduce image size
    processed_frame = processed_frame[::2, ::2, :]
    # remove color
    processed_frame = processed_frame[:, :, 0]
    # remove background
    processed_frame[processed_frame == 144] = 0
    processed_frame[processed_frame == 109] = 0
    # set everything that's not black to white
    processed_frame[processed_frame != 0] = 1
    # flatten frame
    processed_frame = processed_frame.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing
    # on changes in the game
    if previously_processed_frame is not None:
      delta_frame = processed_frame - previously_processed_frame
    else:
      # the below flattens a frame of 0s
      delta_frame = np.zeros(processed_frame.size)
    # store the previous frame so we can subtract from it next time
    previously_processed_frame = processed_frame
    return delta_frame, previously_processed_frame

def discount_rewards(rewards):
    """ Actions you took 20 steps before the end result are less important to the overall
    result than an action you took a step ago. This implements that logic by discounting
    the reward on previous actions based on how long ago they were taken """
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    # earlier frames are discounted a lot more than rewards for later frames
    for i in list(reversed(range(0, rewards.size))):
        if rewards[i] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * discount_rate + rewards[i]
        discounted_rewards[i] = running_add
    return discounted_rewards

def forward(frame):
    hidden_layer = np.dot(weights['W1'], frame)
    hidden_layer[hidden_layer < 0] = 0 # ReLU nonlinearity
    yhat = np.dot(weights['W2'], hidden_layer)
    up_probability = sigmoid(yhat)
    return hidden_layer, up_probability # return probability of taking action 2, and hidden state

def backward(game_hidden_layer_values, game_residuals):
    """ backward pass. (game_hidden_layer_values is array of intermediate hidden states) """
    dW2 = np.dot(game_hidden_layer_values.T, game_residuals).ravel()
    dh = np.outer(game_residuals, weights['W2'])
    dh[game_hidden_layer_values <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, game_delta_frames)
    return {'W1':dW1, 'W2':dW2}

def choose_action(probability):
    """ this is the stochastic part of RL and what makes this policy
    neural network different from a supervised learning problem """
    random_value = np.random.uniform()
    if random_value < probability:
      # signifies up in openai gym
      return 2
    else:
       # signifies down in openai gym
      return 3

def update_weights():
    for key, value in weights.items():
        g = gradient_buffer[key] # gradient
        rmsprop_cache[key] = backprop_rate * rmsprop_cache[key] + (1 - backprop_rate) * g**2
        weights[key] += learning_rate * g / (np.sqrt(rmsprop_cache[key]) + 1e-5)
        gradient_buffer[key] = np.zeros_like(value) # reset batch gradient buffer
    return None

# hyperparameters
num_hidden_layers = 200 # number of hidden layer neurons
games_to_play = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
discount_rate = 0.99 # discount factor for reward
backprop_rate = 0.99 # decay factor for RMSProp leaky sum of weight_gradients^2
resume = True # resume from previous checkpoint?
render = False

# weights initialization
input_layer_size = 80 * 80 # input dimensionality: 80x80 grid
if resume:
    weights = pickle.load(open('nn_weights.pkl', 'rb'))
else:
    weights = {}
    weights['W1'] = np.random.randn(num_hidden_layers,input_layer_size) / np.sqrt(input_layer_size) # "Xavier" initialization
    weights['W2'] = np.random.randn(num_hidden_layers) / np.sqrt(num_hidden_layers)

gradient_buffer = {key : np.zeros_like(value) for key, value in weights.items()} # update buffers that add up gradients over a batch
rmsprop_cache = {key : np.zeros_like(value) for key, value in weights.items()} # rmsprop memory

env = gym.make("Pong-v0")
frame = env.reset()
previously_processed_frame = None # used in computing the difference frame
delta_frames,hidden_layer_values = [], []
residuals, rewards = [], []
running_reward = None
reward_sum = 0
game_number = 0

while True:
  env.render()

  # preprocess the frame, set input to network to be difference image
  delta_frame, previously_processed_frame = process_frame(frame, previously_processed_frame)

  # forward the policy network and sample an action from the returned probability
  hidden_layer, up_probability = forward(delta_frame)
  action = choose_action(up_probability)

  # record various intermediates (needed later for backprop)
  delta_frames.append(delta_frame) #frame
  hidden_layer_values.append(hidden_layer) # hidden state

  # assume every action taken was the true action and
  # calculate gradient
  assumed_true_label = 1 if action == 2 else 0 # a "fake label"
  residuals.append(assumed_true_label - up_probability) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  frame, reward, done, info = env.step(action)
  reward_sum += reward

  rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    game_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    game_delta_frames = np.vstack(delta_frames)
    game_hidden_layer_values = np.vstack(hidden_layer_values)
    game_residuals = np.vstack(residuals)
    game_rewards = np.vstack(rewards)

    # reset array memory
    delta_frames, hidden_layer_values = [], []
    residuals, rewards = [],[]

    # compute the discounted reward backwards through time
    discounted_game_rewards = discount_rewards(game_rewards)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_game_rewards -= np.mean(discounted_game_rewards)
    discounted_game_rewards /= np.std(discounted_game_rewards)
    game_residuals *= discounted_game_rewards # modulate the gradient with advantage (PG magic happens right here.)

    weight_gradients = backward(game_hidden_layer_values, game_residuals)
    for key in weights: gradient_buffer[key] += weight_gradients[key] # accumulate grad over batch

    # perform rmsprop parameter update every games_to_play episodes
    if game_number % games_to_play == 0:
        print("updating weights")
        update_weights()

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('game_number: {0}'.format(game_number))
    print('resetting env. episode reward total was {0}. running mean: {1}'.format(reward_sum, running_reward))
    if game_number % 100 == 0: pickle.dump(weights, open('nn_weights.pkl', 'wb'))
    reward_sum = 0
    frame = env.reset() # reset env
    previously_processed_frame = None

  # if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
  #   print('ep {0}: game finished, reward: {1}'.format(game_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
