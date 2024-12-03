import tkinter as tk
from tkinter import messagebox
import copy
import random
import pickle

# Constants representing the game state values for empty, black, and white pieces.
EMPTY = 0
BLACK = 1
WHITE = 2

# Genetic algorithm parameters, including population size, number of generations, mutation rate, and file for best genome.
POPULATION_SIZE = 20
GENERATIONS = 20
MUTATION_RATE = 0.1
BEST_GENOME_FILE = "best_genome.pkl"

# Board settings, the board dimension can be changed with this variable, it has to be a multiple of 8 to keep the game proportions
BOARD_DIMENSION = 8 # 8x8 is the standard Reversi board size
BOARD_SIZE = BOARD_DIMENSION * BOARD_DIMENSION


class ReversiGame:
    def __init__(self, root, canvas):
        self.root = root # Main window for the tkinter GUI.
        self.canvas = canvas # Canvas for drawing game elements.
        self.clicked_variable = None # Tracks the state of mouse clicks.
        self.canvas.bind("<Button-1>", self.on_click) # Bind left mouse click to the on_click method.
        
        # Initialise the board with empty spaces and set starting positions for black and white discs.
        self.board = [[EMPTY for _ in range(BOARD_DIMENSION)] for _ in range(BOARD_DIMENSION)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.draw_board()
        
        self.current_player = BLACK # Start game with black player
        self.update_status()
        self.player_turn = False # Indicates if it's the player's turn to move.

    def draw_board(self):
        
        self.canvas.delete("discs") # Clear existing discs before redrawing.
        for row in range(BOARD_DIMENSION):
            for col in range(BOARD_DIMENSION):
                # Calculate the top-left and bottom-right coordinates of each square.
                x0, y0 = col * 50, row * 50
                x1, y1 = x0 + 50, y0 + 50
                # Draw a green rectangle for the board square.
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="green", outline="black", tags="discs")
                # Draw black or white ovals for pieces.
                if self.board[row][col] == BLACK:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black", tags="discs")
                elif self.board[row][col] == WHITE:
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="white", tags="discs")

    def update_status(self):
        black_discs = sum(row.count(BLACK) for row in self.board) # Count black discs.
        white_discs = sum(row.count(WHITE) for row in self.board) # Count white discs.
        self.root.title(f"Reversi - Black: {black_discs}, White: {white_discs}")

    def on_click(self, event):
        if self.player_turn: # Only process clicks if it's the player's turn.
            col = event.x // 50 # Determine column by dividing x coordinate by square width.
            row = event.y // 50 # Determine row by dividing y coordinate by square height.
            # Check if the clicked aquare is within the board and empty and if placing a piece here is a valid move..
            if 0 <= row < BOARD_DIMENSION and 0 <= col < BOARD_DIMENSION and self.board[row][col] == EMPTY and self.is_valid_move(row, col):
                self.make_player_move(row, col)

    def make_player_move(self, row, col):
        if self.can_make_move: # Check if any moves are possible at all.
            if self.is_valid_move(row, col): # Validate the specific move.
                self.make_move(row, col) # Make the move.
                self.draw_board() # Redraw the board.
                self.update_status() # Update score display.
                self.player_turn = False  # End player's turn.
                self.current_player = 3 - self.current_player  # Switch players.
                # If it's no longer the player's turn, call the AI to make its move.
                if self.player_turn == False:
                    genetic_algorithm.ai_move(loaded_genome)
        else:
            self.player_turn = False
            self.current_player = 3 - self.current_player
            if self.player_turn == False:
                    genetic_algorithm.ai_move(loaded_genome)
            
            


    def is_valid_move(self, row, col):
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] # All eight possible directions.

        for dr, dc in directions:
            r, c = row + dr, col + dc # Start checking from the adjacent cell in the direction (dr, dc).
            flip_cells = [] # List to keep track of opponent's pieces that could be flipped.
            # Continue moving in the direction until the end of the board or a different piece is encountered.
            while 0 <= r < BOARD_DIMENSION and 0 <= c < BOARD_DIMENSION and self.board[r][c] == 3 - self.current_player:
                flip_cells.append((r, c))
                r, c = r + dr, c + dc
            # Check if the chain of opponent's piece is closed by one of the current player's pieces.
            if 0 <= r < BOARD_DIMENSION and 0 <= c < BOARD_DIMENSION and self.board[r][c] == self.current_player and flip_cells:
                return True
        return False
    
    def can_make_move(self):
        # Check if the current player can make any valid move.  
        for row in range(BOARD_DIMENSION):
            for col in range(BOARD_DIMENSION):
                if self.board[row][col] == EMPTY and self.is_valid_move(row, col):
                    return True
        return False

    def make_move(self, row, col):
        self.board[row][col] = self.current_player # Place the piece on the board.
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            flip_cells = []
            while 0 <= r < BOARD_DIMENSION and 0 <= c < BOARD_DIMENSION and self.board[r][c] == 3 - self.current_player:
                flip_cells.append((r, c))
                r, c = r + dr, c + dc

            if 0 <= r < BOARD_DIMENSION and 0 <= c < BOARD_DIMENSION and self.board[r][c] == self.current_player:
                for fr, fc in flip_cells:
                    self.board[fr][fc] = self.current_player # Flip the opponent's pieces.


class GeneticAlgorithmReversi:
    def __init__(self, game):
        self.game = game # the game instance where the AI will play.
        self.population = self.initialize_population() # Initialise the population with potentially pre-saved best genome.
        self.ai_enabled = True # Flag to enable Ai decisions.

    def initialize_population(self):
        return [self.load_best_genome() for _ in range(POPULATION_SIZE)] # Load the best genome if available, for each individual.

    def create_new_genome(self):
        new_genome = [random.uniform(0, 1) for _ in range(BOARD_SIZE)] # Create a genome with random values for each board position.
        with open(BEST_GENOME_FILE, 'wb') as file:
            pickle.dump(new_genome, file) # Save the new genome to file.

    def evaluate_population(self, ai_game):
        scores = []
        for genome in self.population:
            score = self.play_game(genome, ai_game) # Play a game with each genome and score its performance.
            scores.append(score)
        return scores
    

    def ai_move(self, genome):
        if self.game.current_player == BLACK and not self.game.player_turn:
            # List all valid moves for the AI when it is the black players.
            valid_moves = [(row, col) for row in range(BOARD_DIMENSION) for col in range(BOARD_DIMENSION)
                           if self.game.board[row][col] == EMPTY and self.game.is_valid_move(row, col)]

            if not valid_moves:
                # If no valid moves, switch turn to the player.
                self.game.player_turn = True
                self.game.current_player = 3 - self.game.current_player
                return

            # Evaluate the weight for each valid move using the genome.
            move_weights = [genome[row * BOARD_DIMENSION + col] for row, col in valid_moves]
            # Choose the move with the highest weight.
            max_weight_index = move_weights.index(max(move_weights))
            selected_move = valid_moves[max_weight_index]

            # Execute the chosen move and update the game state.
            self.game.make_move(selected_move[0], selected_move[1])
            self.game.draw_board()  # Update the board
            self.game.update_status()
            self.game.current_player = WHITE  # Switch to the player's turn
            self.game.player_turn = True

        elif self.game.current_player == WHITE and not self.game.player_turn:
            # Repeat similar logic for the white player.
            valid_moves = [(row, col) for row in range(BOARD_DIMENSION) for col in range(BOARD_DIMENSION)
                           if self.game.board[row][col] == EMPTY and self.game.is_valid_move(row, col)]

            if not valid_moves:
                self.game.player_turn = True
                self.game.current_player = 3 - self.game.current_player
                return

            move_weights = [genome[row * BOARD_DIMENSION + col] for row, col in valid_moves]
            max_weight_index = move_weights.index(max(move_weights))
            selected_move = valid_moves[max_weight_index]

            self.game.make_move(selected_move[0], selected_move[1])
            self.game.draw_board()  # Update the board
            self.game.update_status()
            self.game.current_player = BLACK  # Switch to the player's turn
            self.game.player_turn = True
        
    def play_game(self, genome, ai_game):
        current_board = copy.deepcopy(self.game.board) # Store current board state to restore after simulation.
        genome_2d = [genome[i:i + BOARD_DIMENSION] for i in range(0, len(genome), BOARD_DIMENSION)] # Convert genome list to 2D for easier handling.

        while self.can_make_move(): # Continue game until no moves are possible.
            if self.ai_enabled and self.game.current_player == BLACK and not self.game.player_turn:
                # AI plays as black. Logic for move selection is similar to ai_move method
                valid_moves = []
                move_weights = []

                for row in range(BOARD_DIMENSION):
                    for col in range(BOARD_DIMENSION):
                        if self.game.board[row][col] == EMPTY and self.game.is_valid_move(row, col):
                            valid_moves.append((row, col))
                            move_weights.append(genome_2d[row][col])

                if not valid_moves:
                    break # No moves availbable, end the game.

                max_weight_index = move_weights.index(max(move_weights))
                selected_move = valid_moves[max_weight_index]
                self.game.make_move(selected_move[0], selected_move[1])
            elif self.ai_enabled and self.game.current_player == WHITE and not self.game.player_turn:
                # AI plays as white. Selection logic si the same.
                valid_moves = []
                move_weights = []

                for row in range(BOARD_DIMENSION):
                    for col in range(BOARD_DIMENSION):
                        if self.game.board[row][col] == EMPTY and self.game.is_valid_move(row, col):
                            valid_moves.append((row, col))
                            move_weights.append(genome_2d[row][col])

                if not valid_moves:
                    break # No moves available, end the game.

                max_weight_index = move_weights.index(max(move_weights))
                selected_move = valid_moves[max_weight_index]
                self.game.make_move(selected_move[0], selected_move[1])

            self.game.current_player = 3 - self.game.current_player # Switch player after each move.

        final_score = sum(row.count(self.game.current_player) for row in self.game.board)

        self.game.board = current_board # Restore the original board state.
        self.game.current_player = BLACK # Reset player to black.

        if ai_game == False:     
            self.game.player_turn = True # Reset turn to player if not simulating AI game.

        return final_score

    def can_make_move(self):
        for row in range(BOARD_DIMENSION):
            for col in range(BOARD_DIMENSION):
                if self.game.board[row][col] == EMPTY and self.game.is_valid_move(row, col):
                    return True
        return False

    def run_genetic_algorithm(self, ai_game):
        for generation in range(GENERATIONS):
            scores = self.evaluate_population(ai_game) # Evaluate the current population's fitness.
            print(f"Generation {generation + 1}, Max Score: {max(scores)}") # Print the best genome of the current generation.

            # Selection step: Sellect the better-performing genomes to be parents for the next generation.
            selected_population = self.selection(scores)
            # Crossover step: Generate children from selected parents.
            children = self.crossover(selected_population)
            # Mutation step: Apply random mutations to the children to introduce variability.
            new_population = selected_population + children
            new_population = self.mutation(new_population)

            self.population = new_population # Replace the old population with the new one.

        # After all generations, save the best performing genome.
        best_genome = self.population[scores.index(max(scores))]
        print("Best Genome:", best_genome)

        # Save the best genome to a file
        with open(BEST_GENOME_FILE, 'wb') as file:
            pickle.dump(best_genome, file)

        self.play_game(best_genome, ai_game)

    def load_best_genome(self):
        try:
            with open(BEST_GENOME_FILE, 'rb') as file:
                best_genome = pickle.load(file)
            return best_genome
        except FileNotFoundError:
            return None # Return None if no file exists.

    def selection(self, scores):
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        selected_indices = sorted_indices[:int(POPULATION_SIZE / 2)] # Select the top half of individuals.
        return [self.population[i] for i in selected_indices] # Create a list of selected genomes.

    def crossover(self, parents):
        children = []
        for _ in range(POPULATION_SIZE - len(parents)): # Determine how many children to produce.
            parent1, parent2 = random.sample(parents, 2) # Randomly pick two parents.
            crossover_point = random.randint(1, len(parent1) - 1) # Select a random crossover point.
            # Create a new child genome by combining parts of both parent genomes at the crossover point.
            child = parent1[:crossover_point] + parent2[crossover_point:]
            children.append(child)
        return children

    def mutation(self, population):
        for i in range(len(population)):
            for j in range(len(population[i])):
                if random.random() < MUTATION_RATE:
                    # Apply a mutation by adjusting the gene value within a small range.
                    population[i][j] += random.uniform(-0.1, 0.1)
        return population


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Reversi")
    canvas = tk.Canvas(root, width=400, height=400, bg="green")
    canvas.pack()

    game = ReversiGame(root, canvas)
    genetic_algorithm = GeneticAlgorithmReversi(game)
    
    Running = True

    while(Running):
        print("Press 1 to run a training algorithm")
        print("Press 2 to play the game")
        print("Press 3 to exit")
        val = input(":")
        
        if(val == "1"):
            # Load the best genome from the file, if available
            loaded_genome = genetic_algorithm.load_best_genome()
            if loaded_genome is not None:
                print("Loaded Best Genome:", loaded_genome)
                ai_game = True
                genetic_algorithm.run_genetic_algorithm(ai_game)
                ai_game = False
                loaded_genome = genetic_algorithm.load_best_genome()
            else:
                genetic_algorithm.create_new_genome()
        elif(val == "2"):
            ai_game = False
            loaded_genome = genetic_algorithm.load_best_genome()
            if loaded_genome is not None:
                genetic_algorithm.play_game(loaded_genome, ai_game)
                root.mainloop()
            else:
                print("There is no genome, run a training algorithm first")
        elif(val == "3"):
            Running = False
        else:
            print("Invalid input")
            
    


    


