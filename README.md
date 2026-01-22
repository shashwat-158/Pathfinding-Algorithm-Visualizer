# Pathfinding Algorithm Visualizer

An interactive Python application for visualizing and comparing pathfinding algorithms like A*, Dijkstra's, BFS, and Greedy Best-First Search. This tool demonstrates how different algorithms handle weighted graphs versus unweighted exploration.

![Pathfinding Demo](assets/demo.gif)

---

## Key Features

- **Interactive Grid:** Create your own mazes by drawing barrier walls and start/end points.
- **Weighted Nodes (Traffic):** Draw "Traffic" nodes (Brown) that cost more to traverse. This visually demonstrates the intelligence of Dijkstra's algorithm vs. the blindness of BFS.
- **Random Maze Generator:** Instantly generate a complex map with barriers and traffic to test algorithms.
- **Instruction Menu:** Built-in help overlay (press 'I') to view controls without leaving the application.
- **Multiple Algorithms:**
    - **A* (A-Star):** The gold standard for pathfinding; uses heuristics to be fast and accurate.
    - **Dijkstra's Algorithm:** Guarantees the shortest path and intelligently navigates around high-cost traffic.
    - **Breadth-First Search (BFS):** Guarantees the shortest path in unweighted grids but fails to account for traffic costs.
    - **Greedy Best-First Search:** Extremely fast but not guaranteed to find the shortest path.
- **Real-Time Benchmarking:** Live counters for **Nodes Visited** and **Path Length** allow for objective comparison of efficiency.

## Tech Stack

- **Python 3**
- **Pygame**

## Setup and Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/shashwat-158/Heuristic-Search-Algorithm-Analysis-Platform.git](https://github.com/shashwat-158/Heuristic-Search-Algorithm-Analysis-Platform.git)
    ```
2.  Navigate into the project directory:
    ```bash
    cd Heuristic-Search-Algorithm-Analysis-Platform
    ```
3.  Install the required dependencies:
    ```bash
    pip install pygame
    ```

## How to Use

1.  Run the application:
    ```bash
    python main.py
    ```

2.  **Draw the Map:**
    - **Left-Click:** Place **Start** (Orange), then **End** (Purple), then **Barriers** (Black).
    - **Middle-Click (or Hold 'T' + Left-Click):** Draw **Traffic** (Brown). These nodes have a weight of 5, forcing smart algorithms to find a path around them.
    - **Right-Click:** Erase any node.
    - **Press 'M':** Generate a random maze with walls and traffic.

3.  **Run an Algorithm:**
    - **SPACE:** Run A* Algorithm.
    - **D:** Run Dijkstra's Algorithm.
    - **B:** Run Breadth-First Search (BFS).
    - **G:** Run Greedy Best-First Search.

4.  **Controls:**
    - **I:** Toggle the Instructions / Help Menu.
    - **C:** Clear the entire board.
    - **Algorithms automatically clear the previous path** when you run a new one, making it easy to compare results on the same map.