# main.py

# Importing necessary libraries
import pygame
import sys # This lets us close the window
from queue import PriorityQueue

# Initialize Pygame
pygame.init()

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

    # --- Methods for checking the state of the Node ---
    def get_pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == PURPLE
    
    # --- Methods for changing the state of the Node ---
    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_end(self):
        self.color = PURPLE

    def make_barrier(self):
        self.color = BLACK

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


# master function to handle all visual updates
def draw(win, grid, rows, width):
    # wiping window
    win.fill(WHITE)
    # looping through every nodes inside grid
    for row in grid:
        for node in row:
            # telling every node to draw themselves on canvas(win)
            node.draw(win)
    # after drawing all nodes, drawing grid lines on top
    draw_grid(win, rows, width)
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
    # Trace backwards from end to start
    while current in came_from:
        current = came_from[current]
        current.make_path() # Make it TURQUOISE
        draw()

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

        # Check if we have found the destination
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end() # Ensure the end node stays purple
            return True

        # Check all neighbors of the current node
        for neighbor in current.neighbors:
            # Calculate the temporary G-score (current G + distance to neighbor)
            # Distance is always 1 in our grid
            temp_g_score = g_score[current] + 1

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
        draw()

        # If the node we just finished checking is not the start node, 
        # mark it as closed (RED)
        if current != start:
            current.make_closed()

    return False

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

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

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

        draw()

        if current != start:
            current.make_closed()

    return False

# The Main Application Loop
def main():

    # defining number of rows
    ROWS = 50
    # making grid with number of rows and width of window
    grid = make_grid(ROWS, WIDTH)

    # to store start and end of nodes once they are placed
    start = None
    end = None

    run = True
    while run:
        # drawing the grid on screen
        draw(WIN, grid, ROWS, WIDTH)
        # This loop is the "ears" of the program. It listens for user actions.
        for event in pygame.event.get():
            # If the user clicks the 'X' button on the window...
            if event.type == pygame.QUIT:
                run = False # ...stop the main loop.

            if event.type == pygame.KEYDOWN:
                # Common setup for both algorithms
                if start and end:
                     # Update neighbors only when we are ready to run
                    if event.key == pygame.K_SPACE or event.key == pygame.K_d:
                        for row in grid:
                            for node in row:
                                node.update_neighbours(grid)

                    # Press SPACE for A* (A-Star)
                    if event.key == pygame.K_SPACE:
                        algorithm(lambda: draw(WIN, grid, ROWS, WIDTH), grid, start, end)
                    
                    # Press 'D' for Dijkstra
                    if event.key == pygame.K_d:
                        dijkstra(lambda: draw(WIN, grid, ROWS, WIDTH), grid, start, end)

                # Clear screen
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, WIDTH)  
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # mouse click on screen returns tuple(0,0,0) 
                mouse_buttons = pygame.mouse.get_pressed()
                # gives (x,y) coordinate of mouse click
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, WIDTH)
                node = grid[row][col]

                # first value will be 1 on left click
                if mouse_buttons[0]:
                    # Only make it the start node if the end node isn't there yet
                    # AND if the clicked node isn't already the end node.
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

    # Once the loop is finished, quit the program safely.
    pygame.quit()
    sys.exit()

# This line ensures that the main() function runs only when you execute main.py directly
if __name__ == "__main__":
    main()