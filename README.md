# Pathfinding Algorithm Visualizer

An interactive Python application for visualizing and comparing pathfinding algorithms like A*, Dijkstra's, BFS, and Greedy Best-First Search using the Pygame library.

![Screenshot of the pathfinding visualizer in action]
*(A screenshot or GIF will be added here once the project is functional)*

---

## Key Features

- **Interactive Grid:** Create your own mazes by drawing barrier walls with the mouse.
- **Multiple Algorithms:** Visualize how different algorithms "think":
    - **A* (A-Star):** The most efficient pathfinding algorithm using heuristics.
    - **Dijkstra's Algorithm:** Guarantees the shortest path but explores every direction evenly.
    - **Breadth-First Search (BFS):** A fundamental algorithm that guarantees the shortest path in unweighted grids but explores blindly.
    - **Greedy Best-First Search:** A faster, heuristic-heavy algorithm that doesn't always guarantee the shortest path.
- **Real-Time Visualization:** Watch the algorithms explore the grid step-by-step with color-coded nodes (Open, Closed, Path).
- **Performance Benchmarking:** Real-time counters display **Nodes Visited** and **Path Length** to objectively compare the efficiency of each algorithm.
- **Soft Reset:** Automatically clears the previous path when running a new algorithm, allowing for rapid comparison on the same maze.

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

1.  Run the main application:
    ```bash
    python main.py
    ```
2.  **Draw the Grid:**
    - **Left-Click:** Place the **Start Node** (Orange) first, then the **End Node** (Purple), and finally draw **Barrier Walls** (Black).
    - **Right-Click:** Erase any node you click on.

3.  **Run an Algorithm:**
    - Press **SPACE** to run **A* Algorithm**.
    - Press **D** to run **Dijkstra's Algorithm**.
    - Press **G** to run **Greedy Best-First Search**.
    - Press **B** to run **Breadth-First Search**.

4.  **Reset:**
    - Press **C** to clear the entire board and start over.