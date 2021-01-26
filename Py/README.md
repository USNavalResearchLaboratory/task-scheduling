# 6.2 Cognitive Radar Resource Management

## Files

### `main.py` - Assess algorithm performance on a set of scheduling problems
User creates the task scheduling problem by defining:
- the number of tasks and channels
- random generators to draw task sets and channel initialization times
- the number of scheduling problems to generate

User specifies the algorithms with a `list` of `partial` function objects, as well as 
a `list` of Monte Carlo iteration counts per algorithm and a `list` of `str` representations.

