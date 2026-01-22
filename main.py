# main.py

# Importing necessary libraries
import random
import pygame
import sys # This lets us close the window
from queue import PriorityQueue

# Initialize Pygame
pygame.init()
pygame.font.init()
FONT = pygame.font.SysFont('comicsans', 30)

# Setup constants
# This makes it easy to change the window size or colors later
WIDTH, HEIGHT = 800, 800
# surface object returned by .set_mode() stored in WIN
WIN = pygame.display.set_mode((WIDTH, HEIGHT)) # The actual window
pygame.display.set_caption("Pathfinding Algorithm Visualizer") # The title bar text

# Definining color (using RGB values)
WHITE = (255, 255, 255)
# MORE COLORS FOR THE NODES
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0) # Start node
PURPLE = (128, 0, 128)  # End node
BLACK = (0, 0, 0)       # Barrier node
GREY = (128, 128, 128)  # Grid lines
BROWN = (165, 42, 42) # Traffic node


# ------------------- NODE CLASS -------------------
# This is the blueprint for each square on the grid
class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        # These are the actual pixel coordinates on the screen
        self.x = col * width
        self.y = row * width
        self.color = WHITE # Start as a white square
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.weight = 1

    # --- Methods for checking the state of the Node ---
    def get_pos(self):
        return self.row, self.col

    def is_traffic(self):
        return self.color == GREY
    
    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == PURPLE
    
    # --- Methods for changing the state of the Node ---
    def reset(self):
        self.color = WHITE
        self.weight = 1

    def make_start(self):
        self.color = ORANGE
        self.weight = 1

    def make_end(self):
        self.color = PURPLE

    def make_barrier(self):
        self.color = BLACK
        self.weight = 1

    def make_traffic(self):
        self.color = BROWN
        self.weight = 5 # Traffic costs 5x more to move through

    def make_open(self):
        self.color = GREEN # Node is in the "open set"
    
    def make_closed(self):
        self.color = RED # Node has been considered
    
    def make_path(self):
        # A different color for the final path
        self.color = (64, 224, 208) # Turquoise

    # --- The method to draw the Node on the screen ---
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
        self.neighbors = []
        # check DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # Check UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # Check RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        # Check LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

# ----------------------------------------------------


# makes a 2D list of nodes (grid)
# square grids are made because it is better to show algorithms in simple way 
def make_grid(rows, width):
    grid = []
    # size of each square, floor division for whole number
    gap = width//rows
    # outer loop to add list inside list of grid
    for i in range(rows):
        grid.append([])
        # inner loop to create node and add inside current row list
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid

def clear_path(grid):
    for row in grid:
        for node in row:
            # If the node is NOT a start, end, or barrier, reset it to white
            if (node.is_start() or node.is_end() or node.is_barrier()):
                continue
            # If the node has weight > 1, it is traffic. 
            # We restore its color to BROWN.
            if node.weight > 1:
                node.make_traffic()
            else:
                # Otherwise, it's just a normal visited node, so we reset it to WHITE.
                node.reset()

# helper function to draw grid lines
def draw_grid(win, rows, width):
    # calculated size of one square using rows and width of window
    gap = width//rows
    # drawing horizontal lines
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
    # drawing vertical lines
    for j in range(rows):
        pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))

def generate_random_maze(grid, draw):
    # 1. Reset the grid (clearing old walls/traffic)
    for row in grid:
        for node in row:
            if not (node.is_start() or node.is_end()):
                node.reset()

    # 2. Place new items
    for row in grid:
        for node in row:
            # Skip start and end nodes
            if node.is_start() or node.is_end():
                continue

            # Generate a random number between 0.0 and 1.0
            r = random.random()

            # 25% Chance for a Wall (Barrier)
            if r < 0.25:
                node.make_barrier()
            
            # 10% Chance for Traffic (Weight)
            # We check < 0.35 because 0.25 + 0.10 = 0.35
            elif r < 0.35:
                node.make_traffic()
    
    draw()

def draw_instructions(win, width, height):
    # 1. Create a semi-transparent surface
    s = pygame.Surface((width, height))
    s.set_alpha(220) # Transparency level (0-255)
    s.fill(WHITE)
    win.blit(s, (0, 0))

    # 2. Define the text content
    instructions = [
        "--- CONTROLS ---",
        "",
        "[Mouse Controls]",
        "Left Click: Place Start / End / Barriers",
        "Middle Click (or 'T' + Click): Place Traffic (High Cost)",
        "Right Click: Erase Node",
        "",
        "[Algorithms]",
        "SPACE: A* Algorithm",
        "D: Dijkstra's Algorithm",
        "B: Breadth-First Search (BFS)",
        "G: Greedy Best-First Search",
        "",
        "[Grid Options]",
        "M: Generate Random Maze",
        "C: Clear Board",
        "",
        "Press 'I' to Close this Menu"
    ]

    # 3. Render and center the text
    # We use a slightly smaller font for the menu
    menu_font = pygame.font.SysFont('comicsans', 25)
    
    total_height = len(instructions) * 30
    start_y = (height - total_height) // 2

    for i, line in enumerate(instructions):
        # Color the headers (lines starting with --- or []) slightly differently
        color = PURPLE if line.startswith("[") or line.startswith("-") else BLACK
        
        text = menu_font.render(line, 1, color)
        # Center horizontally
        text_rect = text.get_rect(center=(width // 2, start_y + i * 30))
        win.blit(text, text_rect)
# master function to handle all visual updates
# updated draw function to accept 'visited_nodes'
def draw(win, grid, rows, width, visited_nodes=0, path_len=0, algo_name="", show_instructions=False):
    # wiping window
    win.fill(WHITE)
    # looping through every nodes inside grid
    for row in grid:
        for node in row:
            # telling every node to draw themselves on canvas(win)
            node.draw(win)
    # after drawing all nodes, drawing grid lines on top
    draw_grid(win, rows, width)

    # Text 1: Nodes Visited
    text_visited = FONT.render(f"Nodes Visited: {visited_nodes}", 1, PURPLE)
    win.blit(text_visited, (10, 10))

    # Text 2: Path Length
    text_path = FONT.render(f"Path Length: {path_len}", 1, PURPLE)
    win.blit(text_path, (10, 35)) # Draw it slightly below the first text

    # Text 3: Algorithm Name
    text_algo = FONT.render(f"Algorithm: {algo_name}", 1, ORANGE)
    win.blit(text_algo, (10, 60)) # Draw it below Path Length

    # Instruction Hint
    text_hint = FONT.render("Press 'I' for Controls", 1, GREY)
    # Draw it at the bottom right
    win.blit(text_hint, (width - text_hint.get_width() - 10, 10))

    # NEW: Draw the overlay if requested
    if show_instructions:
        draw_instructions(win, width, width) # Assuming square window (width=height)

    # updating the screen to show the result
    pygame.display.update()

# helper function to give mouse position as rows and cols to grid
def get_clicked_pos(pos, rows, width):
    # calculated size of one square using rows and width of window
    gap = width//rows
    # y coordinate divided by size of square will tell it's inside which row
    row = pos[1]//gap
    # x coordinate divided by size of square will tell it's inside which col
    col = pos[0]//gap

    return row, col

# small helper function for manhattan distance calculation in algorithm
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
    path_len = 0
    # Trace backwards from end to start
    while current in came_from:
        current = came_from[current]
        current.make_path() # Make it TURQUOISE
        path_len += 1
        draw()
    return path_len

def algorithm(draw, grid, start, end):
    # We pass 'draw' as a function so we can call it to update the visuals
    # count is used to handle cases where two nodes have the same F-score
    # one added first will get a lower count and be prioritized
    count = 0
    # This store the nodes we need to consider
    open_set = PriorityQueue()
    # We are putting a tuple in the queue: (f_score, count, node)
    # The queue automatically sorts by the first item (the F-score)
    open_set.put((0, count, start))
    # this dictionary act as map to reconstruct the final path
    came_from = {}
    # dictionaries are used to store scores of each node
    # every node's score is initialized to infinity
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    f_score = {node: float("inf") for row in grid for node in row}
    # F-score for start is purely the heuristic (distance to end)
    f_score[start] = h(start.get_pos(), end.get_pos())

    # Create a set to keep track of items in the priority queue (for O(1) lookup)
    open_set_hash = {start}

    nodes_visited = 0

    while not open_set.empty():
        # Allow the user to quit even while the algorithm is running
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get the node with the lowest F-score (highest priority)
        # .get() returns (f_score, count, node), so we take index [2]
        current = open_set.get()[2]
        # Sync our hash set
        open_set_hash.remove(current)

        nodes_visited += 1

        # Check if we have found the destination
        if current == end:
            path_len = reconstruct_path(came_from, end, draw)
            end.make_end() # Ensure the end node stays purple
            return True, nodes_visited, path_len

        # Check all neighbors of the current node
        for neighbor in current.neighbors:
            # Calculate the temporary G-score (current G + distance to neighbor)
            # The cost to move to the neighbor is the neighbor's weight
            temp_g_score = g_score[current] + neighbor.weight

            # If this new path is shorter than the previous known path to this neighbor
            if temp_g_score < g_score[neighbor]:
                # Update the path and scores
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())

                # If the neighbor is not already in the queue to be explored, add it
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open() # Make it GREEN (open set)

        # Update the visual grid
        draw(visited_nodes=nodes_visited)

        # If the node we just finished checking is not the start node, 
        # mark it as closed (RED)
        if current != start:
            current.make_closed()

    return False, 0, 0

def dijkstra(draw, grid, start, end):
    # Dijkstra is identical to A* but without the heuristic (h-score)
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    
    # We only need G-score (distance from start)
    # Initialize to infinity
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    open_set_hash = {start}

    nodes_visited = 0

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        nodes_visited += 1

        if current == end:
            path_len = reconstruct_path(came_from, end, draw)
            end.make_end()
            return True, nodes_visited, path_len

        for neighbor in current.neighbors:
            # The cost to move to the neighbor is the neighbor's weight
            temp_g_score = g_score[current] + neighbor.weight

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                # F-score is just the G-score (no H-score)
                # We still use the Priority Queue to always explore the shortest path found so far
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw(visited_nodes=nodes_visited)

        if current != start:
            current.make_closed()

    return False, 0, 0

def greedy_bfs(draw, grid, start, end):
    # GREEDY IGNORES WEIGHTS (Only cares about H-score)
    count = 0
    open_set = PriorityQueue()
    # Priority is purely the H-score (heuristic)
    open_set.put((0, count, start))
    came_from = {}
    
    # We still keep track of visited nodes to avoid infinite loops, 
    # but we don't strictly need G-score for sorting. 
    # We just need a set to keep track of what we've added.
    open_set_hash = {start}

    nodes_visited = 0

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        nodes_visited += 1

        if current == end:
            path_len = reconstruct_path(came_from, end, draw)
            end.make_end()
            return True, nodes_visited, path_len

        for neighbor in current.neighbors:
            # If we haven't visited this neighbor yet
            if neighbor not in came_from and neighbor != start:
                came_from[neighbor] = current
                
                if neighbor not in open_set_hash:
                    count += 1
                    # Score is ONLY the Heuristic
                    score = h(neighbor.get_pos(), end.get_pos())
                    open_set.put((score, count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw(visited_nodes=nodes_visited)

        if current != start:
            current.make_closed()

    return False, 0, 0

def bfs(draw, grid, start, end):
    # BFS IGNORES WEIGHTS (Treats everything as 1 step)
    # BFS uses a simple Queue (FIFO) logic. 
    # We can use a list and always take the first element (pop(0)).
    queue = [start]
    
    # We need to keep track of visited nodes so we don't process them twice
    visited = {start}
    came_from = {}

    nodes_visited = 0

    while len(queue) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Pop the first element from the list (First-In, First-Out)
        current = queue.pop(0)

        nodes_visited += 1

        if current == end:
            path_len = reconstruct_path(came_from, end, draw)
            end.make_end()
            return True, nodes_visited, path_len

        for neighbor in current.neighbors:
            # Simple logic: If we haven't seen it, add it to the queue
            if neighbor not in visited:
                came_from[neighbor] = current
                visited.add(neighbor)
                queue.append(neighbor)
                neighbor.make_open()

        draw(visited_nodes=nodes_visited)

        if current != start:
            current.make_closed()

    return False, 0, 0

# The Main Application Loop
def main():

    # defining number of rows
    ROWS = 50
    # making grid with number of rows and width of window
    grid = make_grid(ROWS, WIDTH)

    # to store start and end of nodes once they are placed
    start = None
    end = None
    visited_nodes = 0
    path_len = 0
    algo_name = "Pick an Algorithm" # Default text

    # state variable for the menu
    show_instructions = True # Show it by default on start

    run = True
    while run:
        # drawing the grid on screen
        draw(WIN, grid, ROWS, WIDTH, visited_nodes, path_len, algo_name, show_instructions)
        # This loop is the "ears" of the program. It listens for user actions.
        for event in pygame.event.get():
            # If the user clicks the 'X' button on the window...
            if event.type == pygame.QUIT:
                run = False # ...stop the main loop.

            if event.type == pygame.KEYDOWN:
                # toggle Instructions
                if event.key == pygame.K_i:
                    show_instructions = not show_instructions

                # Only allow other inputs if the menu is CLOSED
                if not show_instructions:
                    if start and end:
                        # Update neighbors for all algorithms
                        if event.key in [pygame.K_SPACE, pygame.K_d, pygame.K_g, pygame.K_b]:
                            clear_path(grid)
                            for row in grid:
                                for node in row:
                                    node.update_neighbours(grid)

                        # SPACE = A*
                        if event.key == pygame.K_SPACE:
                            algo_name = "A* Algorithm"
                            _, visited_nodes, path_len = algorithm(
                                lambda visited_nodes=visited_nodes, path_len=path_len, algo_name=algo_name: 
                                draw(WIN, grid, ROWS, WIDTH, visited_nodes, path_len, algo_name),
                                grid, start, end
                            )
                        
                        # D = Dijkstra
                        if event.key == pygame.K_d:
                            algo_name = "Dijkstra's Algorithm"
                            _, visited_nodes, path_len = dijkstra(
                                lambda visited_nodes=visited_nodes, path_len=path_len, algo_name=algo_name: 
                                draw(WIN, grid, ROWS, WIDTH, visited_nodes, path_len, algo_name),
                                grid, start, end
                            )

                        # G = Greedy BFS
                        if event.key == pygame.K_g:
                            algo_name = "Greedy Best-First Search"
                            _, visited_nodes, path_len = greedy_bfs(
                                lambda visited_nodes=visited_nodes, path_len=path_len, algo_name=algo_name: 
                                draw(WIN, grid, ROWS, WIDTH, visited_nodes, path_len, algo_name),
                                grid, start, end
                            )

                        # B = Breadth-First Search
                        if event.key == pygame.K_b:
                            algo_name = "Breadth-First Search"
                            _, visited_nodes, path_len = bfs(
                                lambda visited_nodes=visited_nodes, path_len=path_len, algo_name=algo_name: 
                                draw(WIN, grid, ROWS, WIDTH, visited_nodes, path_len, algo_name),
                                grid, start, end
                            )

                    # Clear screen ('C')
                    if event.key == pygame.K_c:
                        start = None
                        end = None
                        grid = make_grid(ROWS, WIDTH)
                        visited_nodes = 0
                        path_len = 0
                        algo_name = "Pick an Algorithm"
                    
                    # Generate Random Maze ('M')
                    if event.key == pygame.K_m:
                        # Reset stats so the new maze looks clean
                        visited_nodes = 0
                        path_len = 0
                        algo_name = "Pick an Algorithm"
                        generate_random_maze(grid, lambda: draw(WIN, grid, ROWS, WIDTH, visited_nodes, path_len, algo_name))
            
            # Only allow mouse clicks if menu is CLOSED
            if not show_instructions and event.type == pygame.MOUSEBUTTONDOWN:
                # mouse click on screen returns tuple(0,0,0) 
                mouse_buttons = pygame.mouse.get_pressed()
                # gives (x,y) coordinate of mouse click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, WIDTH)
                node = grid[row][col]
                keys = pygame.key.get_pressed() # Check keyboard state

                # first value will be 1 on left click
                if mouse_buttons[0]:
                    # Option 1: Holding 'T' + Left Click -> Make Traffic
                    if keys[pygame.K_t]:
                         if not node.is_start() and not node.is_end() and not node.is_barrier():
                            node.make_traffic()
                    # Only make it the start node if the end node isn't there yet
                    # AND if the clicked node isn't already the end node.
                    else:
                        if not start and node != end:
                            start = node
                            start.make_start()
                        # Only make it the end node if the start node already exists
                        # AND if the clicked node isn't already the start node.
                        elif not end and node != start:
                            end = node
                            end.make_end()
                        # Only make it a barrier if it's not the start or end node.
                        elif node != end and node != start:
                            node.make_barrier()
                # If right mouse button was clicked
                elif mouse_buttons[2]:
                    # This is where you'll add the logic to reset nodes
                    node.reset()
                    if node == start:
                        start = None
                    elif node == end:
                        end = None

                # Actually, Pygame mouse_buttons[1] is the Middle Click (Scroll wheel click)
                elif mouse_buttons[1]: 
                    if not node.is_start() and not node.is_end() and not node.is_barrier():
                        node.make_traffic()

    # Once the loop is finished, quit the program safely.
    pygame.quit()
    sys.exit()

# This line ensures that the main() function runs only when you execute main.py directly
if __name__ == "__main__":
    main()