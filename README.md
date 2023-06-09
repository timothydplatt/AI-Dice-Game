# Assignment 2 - Dice Game Agent

## Context
The challenge was to implement an agent that is able to achieve a high score playing a dice game. The dice game works as follows:

- The agent starts with 0 points.
- Three fair six-sided dice are rolled to provide a score i.e. 2 + 4 + 6 = 12. 
- If two or more die show the same values e.g. 1 + 1 + 4 then all matching value die are flipped upside down to become 6 + 6 + 4 = 16.
- After the initial dice roll the agent playing has two options:
  - Stick and accept the values presented by holding all 3 die.
  - Reroll any combination of the dice i.e. holding none, one or two of the die. 
- Rerolling dice costs the agent 1 point, which means negative scores are possible. 

The game can be represented as a Makov Decision Process as it has the following components:

- A set (S) of 56 states (𝑠).
- A set (A) of 8 possible actions (a).
  - A policy π(s) is the action (a) that is to be performed given a state (𝑠).
- Transition probabilities P(s<sup>1</sup> | s, a) which are the probability that an action (a) in state (𝑠) at time (t) will lead to state s<sup>1</sup> at time t+1.
- A reward function R(s) which is the reward after transitioning from one state (𝑠) to another (s<sup>1</sup>), which is -1 for a 'reroll' action or the value of the dice for a 'stick' action.

## Approach
As suggested in the assignment I decided to implement the value iteration algorithm to produce a dice game strategy for the agent and performed parameter optimisation to predetermine the best parameters to use to maximise the score achieved by the agent. 

I first started out by playing the game several times and printing out different properties and function output to understand how actions, states, rewards and probabilities etc were all represented. Additionally, I also studied the code that was already in place to familiarise myself before diving into building a solution - it was a worthwhile endeavour and I noticed little quirks which may have confused me otherwise e.g. the states of the game are always sorted in ascending order. 

### Value Iteration
After selections some sensible default values for my discount_factor (𝛾) and theta (θ) I immediately jumped into trying to translate the value iteration function psuedo code from our course material and the modules recommended reading. 


```
XXX
```
- Understanding/interpreting the Bellman update.
- Deciding how to handle is game over. 
- Nested loops vs. breaking the function out into separate functions (AIMA much more complex, felt like happy medium)

### Parameter Optimisation

```
XXX
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

The Markov reward process (MRP) is defined by (S, P, R, γ), 

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
