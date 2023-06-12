# Assignment 2 - Dice Game Agent

## Context
The challenge was to implement an agent able to achieve an optimal score when playing a dice game. The dice game works as follows:

- The agent starts with 0 points.
- Three fair six-sided dice are rolled to provide a score i.e. 2 + 4 + 6 = 12. 
- If two or more die show the same values e.g. 1 + 1 + 4 then all matching value die are flipped upside down to become 6 + 6 + 4 = 16.
- After the initial dice roll the agent has two options:
  - Stick and accept the values presented by holding all 3 die.
  - Reroll any combination of the die i.e. holding none, one or two of the die. 
- Rerolling dice costs the agent 1 point, which means negative scores are possible. 

The game can be represented as a Makov Decision Process as it has the following components:

- A set (S) of 56 states (𝑠).
- A set (A) of 8 possible actions (a).
  - A policy π(s) is the action (a) that is to be performed given a state (𝑠).
- Transition probabilities P(s<sup>1</sup> | s, a) which are the probability that an action (a) in state (𝑠) at time (t) will lead to state s<sup>1</sup> at time t+1.
- A reward function R(s) which is the reward after transitioning from one state (𝑠) to another (s<sup>1</sup>), which is -1 for a 'reroll' action or the value of the dice for a 'stick' action.

## Approach
As suggested in the assignment I decided to implement the value iteration algorithm to produce a dice game strategy for the agent and performed parameter optimisation to predetermine the best parameters to use to maximise the score achieved by the agent while ensuring the agents performance remained efficient. 

I first started out by playing the game several times and printing out different properties and output of the various functions to understand how actions, states, rewards and probabilities etc were represented. Additionally, I studied the code that was already in place to familiarise myself before diving into coding a solution - it was a worthwhile endeavour and I noticed several little quirks which may have confused me had I not studied the code first e.g. the states of the game are always sorted in ascending order. 

### Value Iteration
After selecting some sensible default values for my discount_factor (𝛾) and theta (θ)* I immediately moved on to trying to translate the value iteration algorithm psuedo code from our course material** and the modules recommended reading into actual code (Russell et al., 2010). The value iteration algorithm presented in the course material is as follows:

<p align="center">
  <img src="https://github.com/timothydplatt/AI-Dice-Game/blob/main/Lecture.png" width=50% height=50%>
</p>

There are numerous ways this code could be implemented. Artificial Intelligence: A Modern Approach*** is quite succict but not particularly readable in my opinion - it's also the only example that uses epsilon versus theta to represent a small value to determine if the algorithm has converged, which is the greek symbol I've traditionally understood to represent a small arbitrary value so it was interesting to see all other texts use theta to represent a small arbitrary value. 

Some techniques break the value iteration algorithm down into multiple functions**** - usually having a sepearate "one-step look-ahead" function, which performs the Bellman update, to break out this quite complex calculation into discrete operations. Typically, I'd find this a more intuitive approach than having several nested loops all within a single function but in the case of value iteration I find keeping the code all in one function more readable and easier to understand what's happening. Equally, the results of the operation are stored in all manner of ways, with some examples storing the the state/values in one dictionary and one state/policy in another dictionary or using Q tables to represent states and rewards in a tabular format*****. 

My implementation of the value iteration function, with comments removed, is shown below:

``` python
def value_iteration(self):
        V = {state: [None, 0] for state in game.states}
        
        while True:
            delta = 0
            for state in game.states:
                current_state_value = V[state][1]
                best_action_value = 0

                for action in game.actions:
                    expected_return = 0                 
                    next_states, game_over, reward, probabilities = game.get_next_states(action, state)
                    state_probability_iterable = zip(next_states, probabilities)
                    
                    for next_state, probability in state_probability_iterable:
                        if not game_over:
                            expected_return += probability * (reward + self.discount_factor * V[next_state][1])
                        else:
                            expected_return += probability * reward

                    if expected_return > best_action_value:
                        best_action_value = expected_return
                        V[state] = [action, best_action_value]
                
                delta = max(delta, abs(current_state_value - best_action_value))
                
            if delta < self.theta:
                break
    
        return V
```

The variable V is a dictionary where the key is the different states of the game and the value is a list containing an action in index 0 and the expected value of perorming that action for a given state in index 1. I chose the value of the dictionary to be a list as opposed to a tuple because I was expected to be updating the values of the action/value for each state at different times and this wouldn't be possible (or simple) given tuples are immutable. In the end I went on to update the values of the dictionary at the same time so a tuple would have perhaps been a good idea given they're slightly more performant than a list. As mentioned, Q tables can also be used but it didn't feel necessary to import a data collection outside of the standard Python language.

The most challenging part of the value function was writing the Bellman update:

``` python
for next_state, probability in state_probability_iterable:
  if not game_over:
      expected_return += probability * (reward + self.discount_factor * V[next_state][1])
  else:
      expected_return += probability * reward
```

Firstly, it's a complex function to translate from mathematical notation to code - it's important to get the operands and order of operations correct - I managed to create both an infinite loop and a poor scoring agent by getting these wrong and it took some trial and error. Equally, I initially failed to handle when game_over = true which caused issues as there was no 'next-state'.

XXX


* Save scores and actions in different diaries.
* Implementing the Bellman equation.

- Understanding/interpreting the Bellman update.
- Deciding how to handle is game over. 
- Nested loops vs. breaking the function out into separate functions (AIMA much more complex, felt like happy medium)

### Parameter Optimisation

``` python
def parameter_optimisation():
    game_scores = []
    game_times = []
    possible_theta_values = []
    possible_discount_factor_values = []
    all_combinations_of_discount_factor_and_theta = []
    
    for i in range(-10, 0):
        possible_theta_values.append(10**i)
    possible_theta_values.reverse()

    possible_discount_factor_values = np.arange(0.9, 1.01, 0.01)
    
    for i in possible_discount_factor_values:
        for j in possible_theta_values:
            all_combinations_of_discount_factor_and_theta.append((i, j))
    
    for combination in all_combinations_of_discount_factor_and_theta:
        total_score = 0
        total_time = 0
        n = 1

        np.random.seed()
        game = DiceGame()

        start_time = time.process_time()
        print(round(combination[0], 2))
        print(combination[1])
        test_agent = MyAgent(game, discount_factor=combination[0], theta=combination[1]) #TODO - pass in the values of gamma and theta
        total_time += time.process_time() - start_time

        for ni in range(n):
            start_time = time.process_time()
            score = play_game_with_agent(test_agent, game)
            total_time += time.process_time() - start_time
            total_score += score

        game_scores.append(total_score/n)
        game_times.append(total_time)

    for combination, comb_score, comb_time in zip(all_combinations_of_discount_factor_and_theta, game_scores, game_times):
```

- Tested discount rates of 0.9 and upwards.
- Tested theta values of 0.1 and below.
- Ran game 50,000 times for each combination of those to parameters.
- Discount rate has a strong positive correlation with increased score.
- Theta values didn't have any material impact on scores.
- A higher discount rate means more iterations, which means a longer processing, so measuring time was an important factor. 

## Results

## Reflections
- What could be done to improve the solution?
- What other approaches could I have taken?

## References

Russell, S.J. and Norvig, P. (2010). Artificial intelligence : a modern approach. Upper Saddle River: Prentice-Hall.

* https://engage.bath.ac.uk/learn/mod/forum/discuss.php?d=72269#p159024
‌** https://engage.bath.ac.uk/learn/mod/page/view.php?id=190322
*** https://github.com/aimacode/aima-python/blob/master/mdp.py
**** https://www.baeldung.com/cs/ml-value-iteration-vs-policy-iteration 
***** https://gibberblot.github.io/rl-notes/single-agent/value-iteration.html



XXXXXXXXXX


 So, we’re able to use a one-step look-ahead approach and compute rewards for all possible actions.

A text file that explains your approach and the decisions you made in your own words – a readme file
Submissions that do not include the written section will receive zero marks – this part is mandatory
You may write your file in plain text (.txt) or Markdown (.md)
To get top marks on this assignment, as well as getting a high grade from your implementation, you must also demonstrate excellent academic presentation in your written section

 This means that the maximum initial error ||U0 − U|| ≤ 2Rmax/(1 − γ).
Suppose we run for N iterations to reach an error of at most . Then, because the error is
reduced by at least γ each time, we require γN · 2Rmax/(1 − γ) ≤ . Taking logs, we find
N = (log(2Rmax/(1 − γ))/ log(1/γ))iterations suffice. Figure 17.5(b) shows how N varies with γ, for different values of the ratio
/Rmax. The good news is that, because of the exponentially fast convergence, N does not
depend much on the ratio /Rmax. The bad news is that N grows rapidly as γ becomes close
to 1. We can get fast convergence if we make γ small, but this effectively gives the agent a
short horizon and could miss the long-term effects of the agent’s actions.

- A discount factor (𝛾)
