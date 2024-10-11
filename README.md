# ConnectX Agent - README

## Overview
This repository presents a custom agent designed for the ConnectX competition on Kaggle, employing the **Minimax algorithm enhanced with Alpha-Beta Pruning**. This agent is capable of making sophisticated strategic decisions by heuristically evaluating board states, thereby ensuring competitive gameplay.

## Agent Description
The agent leverages the **Minimax algorithm with Alpha-Beta Pruning** to evaluate potential moves and select the optimal action. By anticipating the opponent's responses, the agent is able to maximize its probability of victory while minimizing the potential for adversarial success.

- **Minimax with Alpha-Beta Pruning**: This method enables the efficient exploration of the game tree by eliminating branches that do not require evaluation, thereby reducing computational overhead and focusing resources on the most promising move sequences.

## How the Agent Works
### Minimax with Alpha-Beta Pruning
The agent utilizes **Minimax with Alpha-Beta Pruning** to determine the optimal move by simulating future game states and assessing the expected outcomes.

- **Evaluation Function**: A heuristic evaluation function estimates the utility of board positions by analyzing winning opportunities, threats posed by the opponent, and control of the center column.
- **Alpha-Beta Pruning**: This technique enhances computational efficiency by pruning branches that cannot affect the final outcome, thus streamlining the decision-making process.

### Heuristic Evaluation
The heuristic evaluation function assigns scores to board configurations based on several critical factors:
- **Center Control**: Dominance of the center column is prioritized, as it provides greater flexibility and opportunities to create winning sequences.
- **Horizontal, Vertical, and Diagonal Threats**: The agent assesses potential winning sequences in multiple directions, assigning scores based on the number of consecutive pieces, with a focus on maximizing its own opportunities while mitigating those of the opponent.

## Code Structure
- **Minimax Implementation**: The `minimax()` function implements the Minimax algorithm with alpha-beta pruning, enabling the agent to evaluate game states and determine the most favorable move.
- **Helper Functions**: A suite of helper functions is provided to support tasks such as validating moves, dropping pieces, evaluating board positions, and determining winning conditions.

## Parameters and Heuristics
- **Depth Adjustment**: The search depth for the Minimax algorithm is set to 4 by default but can be adjusted dynamically based on the complexity of the game state and available computational resources.
- **Evaluation Heuristics**: The evaluation function scores board states based on factors such as the number of consecutive pieces, threats from the opponent, and control of key strategic areas like the center column.

## Usage
The agent can be used within the ConnectX environment by initializing it as follows:

```python
from kaggle_environments import make

env = make("connectx", debug=True)
env.run([agent, "random"])
env.render(mode="ipython")
```

## Results
The agent achieved a score of **904** and attained **7th place** in the ConnectX competition on Kaggle, demonstrating its proficiency in accurately evaluating board positions and making optimal decisions in real time.

## Future Improvements
- **Neural Network Integration**: Incorporate a neural network to improve board state evaluations and reduce reliance on heuristic rules.
- **Adaptive Depth**: Adapt the depth of the Minimax search based on remaining computation time, allowing a more dynamic balance between exploration and optimal decision-making.
- **Enhanced Heuristics**: Refine the heuristic evaluation function to include more advanced features, such as early blocking of opponent threats and the assessment of more complex board patterns.

## Acknowledgements
This agent draws inspiration from established AI methodologies for game playing, particularly the Minimax algorithm with Alpha-Beta Pruning, which has been successfully utilized in traditional games like chess and tic-tac-toe.

## License
This project is licensed under the MIT License.
