# Assignment 2 - Dice Game Agent

## Context
- 56 states
- 8 possible actions

## Approach
As suggested in the assignment I decided to implement the value iteration algorithm to produce a dice game strategy for the agent and performed parameter optimisation to predetermine the best parameters to use to maximise the score achieved by the agent. 

### Value Iteration

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
