import tkinter as tk
import copy

SIZE = 6  # Дошка 6х6

class CheckersProGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Класичні шашки: Ланцюжкове биття працює!")
        self.root.geometry("420x530")
        self.root.config(bg="#2c3e50")

        self.board = [
            ["", "c", "", "c", "", "c"],
            ["c", "", "c", "", "c", ""],
            ["", "", "", "", "", ""],
            ["", "", "", "", "", ""],
            ["", "p", "", "p", "", "p"],
            ["p", "", "p", "", "p", ""]
        ]

        self.turn = "p"             # "p" — хід гравця, "c" — комп'ютер
        self.selected_piece = None  # (r, c) обраної шашки
        self.valid_moves = []       # Список можливих ходів
        self.active_jump_piece = None # Фіксація фішки для подвійного/потрійного биття

        self.label = tk.Label(root, text="Ваш хід! Оберіть шашку", font=("Segoe UI", 12, "bold"), bg="#2c3e50", fg="white")
        self.label.pack(pady=10)

        self.frame = tk.Frame(root, bg="#34495e", bd=4, relief="ridge")
        self.frame.pack(pady=5)

        self.buttons = []
        for r in range(SIZE):
            row_btns = []
            for c in range(SIZE):
                is_dark = (r + c) % 2 == 1
                bg_color = "#7f8c8d" if is_dark else "#ecf0f1"
                btn = tk.Button(self.frame, text="", font=("Segoe UI", 20, "bold"), width=3, height=1,
                                bg=bg_color, fg="#2c3e50", activebackground="#bdc3c7",
                                command=lambda row=r, col=c: self.on_cell_click(row, col))
                btn.grid(row=r, column=c, padx=1, pady=1)
                row_btns.append(btn)
            self.buttons.append(row_btns)

        self.reset_btn = tk.Button(root, text="Нова гра", font=("Segoe UI", 11, "bold"), bg="#e74c3c", fg="white",
                                   command=self.reset_game, relief="flat", padx=15, pady=6)
        self.reset_btn.pack(pady=12)

        self.update_ui()

    def update_ui(self):
        for r in range(SIZE):
            for c in range(SIZE):
                val = self.board[r][c]
                text = ""
                if val == "p": text = "⚪"
                elif val == "P": text = "👑"
                elif val == "c": text = "⚫"
                elif val == "C": text = "♚"

                is_dark = (r + c) % 2 == 1
                bg = "#7f8c8d" if is_dark else "#ecf0f1"

                if self.selected_piece == (r, c):
                    bg = "#f39c12"
                elif any(move['end'] == (r, c) for move in self.valid_moves):
                    bg = "#27ae60"

                self.buttons[r][c].config(text=text, bg=bg)

    def get_piece_moves(self, board, r, c, must_jump_only=False):
        piece = board[r][c]
        if piece == "":
            return []

        moves = []
        is_king = piece.isupper()
        is_player = piece in ('p', 'P')
        opponents = ('c', 'C') if is_player else ('p', 'P')

        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            if not is_king:
                # Звичайний хід (якщо не вимагається обов'язкове биття)
                if not must_jump_only:
                    if (is_player and dr == -1) or (not is_player and dr == 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == "":
                            moves.append({'start': (r, c), 'end': (nr, nc), 'jump': None})
                
                # Биття звичайною шашкою
                nr, nc = r + dr, c + dc
                nnr, nnc = r + 2*dr, c + 2*dc
                if 0 <= nnr < SIZE and 0 <= nnc < SIZE:
                    if board[nr][nc] in opponents and board[nnr][nnc] == "":
                        moves.append({'start': (r, c), 'end': (nnr, nnc), 'jump': (nr, nc)})
            else:
                # Летюча дамка
                found_opponent = None
                nr, nc = r + dr, c + dc
                while 0 <= nr < SIZE and 0 <= nc < SIZE:
                    cell = board[nr][nc]
                    if cell == "":
                        if not found_opponent:
                            if not must_jump_only:
                                moves.append({'start': (r, c), 'end': (nr, nc), 'jump': None})
                        else:
                            moves.append({'start': (r, c), 'end': (nr, nc), 'jump': found_opponent})
                    else:
                        if cell in opponents:
                            if found_opponent is None:
                                found_opponent = (nr, nc)
                            else:
                                break 
                        else:
                            break 
                    nr += dr
                    nc += dc

        return moves

    def get_all_moves(self, board, player_turn):
        pieces_chars = ('p', 'P') if player_turn == 'p' else ('c', 'C')
        all_moves = []
        
        # Якщо триває серія бить однією фішкою, шукаємо ходи ТІЛЬКИ для неї і ТІЛЬКИ биття
        if self.active_jump_piece:
            r, c = self.active_jump_piece
            if board[r][c] in pieces_chars:
                return [m for m in self.get_piece_moves(board, r, c, must_jump_only=True) if m['jump'] is not None]
            return []

        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] in pieces_chars:
                    all_moves.extend(self.get_piece_moves(board, r, c))

        # Правило обов'язкового биття глобально
        jump_moves = [m for m in all_moves if m['jump'] is not None]
        if jump_moves:
            return jump_moves

        return all_moves

    def on_cell_click(self, r, c):
        if self.turn != "p":
            return

        piece = self.board[r][c]

        # Якщо йде ланцюжкове биття, гравець зобов'язаний клікати лише на активну фішку
        if self.active_jump_piece:
            if (r, c) == self.active_jump_piece:
                return # Вона вже обрана
            elif self.selected_piece == self.active_jump_piece:
                # Перевіряємо чи клікнули на куди походити цією фішкою
                matched_move = next((m for m in self.valid_moves if m['end'] == (r, c)), None)
                if matched_move:
                    self.execute_move(matched_move)
                return
            return

        # Звичайний вибір шашки
        if piece in ('p', 'P'):
            all_allowed_moves = self.get_all_moves(self.board, 'p')
            must_jump = any(m['jump'] is not None for m in all_allowed_moves)
            
            piece_moves = self.get_piece_moves(self.board, r, c, must_jump_only=must_jump)
            allowed_piece_moves = [m for m in piece_moves if m in all_allowed_moves]

            if allowed_piece_moves:
                self.selected_piece = (r, c)
                self.valid_moves = allowed_piece_moves
                self.update_ui()
            else:
                self.selected_piece = None
                self.valid_moves = []
                self.update_ui()
                if must_jump:
                    self.label.config(text="Є обов'язкове биття! Оберіть правильну шашку.")
            return

        # Клік на вільну клітинку для звичайного ходу
        if self.selected_piece:
            matched_move = next((m for m in self.valid_moves if m['end'] == (r, c)), None)
            if matched_move:
                self.execute_move(matched_move)

    def execute_move(self, move):
        self.make_move(self.board, move)
        is_jump = move['jump'] is not None
        er, ec = move['end']

        # Перетворення у дамку (якщо звичайна шашка дійшла до краю)
        if er == 0 and self.board[er][ec] == 'p':
            self.board[er][ec] = 'P'
            is_jump = False # Дамка зупиняє подальше биття у цей же хід за правилами

        # Перевірка на продовження ланцюжкового биття
        if is_jump:
            next_jumps = [m for m in self.get_piece_moves(self.board, er, ec, must_jump_only=True) if m['jump'] is not None]
            if next_jumps:
                self.active_jump_piece = (er, ec)
                self.selected_piece = (er, ec)
                self.valid_moves = next_jumps
                self.update_ui()
                self.label.config(text="Подвійне биття! Оберіть наступну ціль цією ж фішкою.")
                return

        # Хід завершено повністю
        self.active_jump_piece = None
        self.selected_piece = None
        self.valid_moves = []
        self.update_ui()

        if not self.get_all_moves(self.board, 'c'):
            self.end_game("Вітаю! Ви перемогли! 🎉")
            return

        self.turn = "c"
        self.label.config(text="Думає комп'ютер... 🤖")
        self.root.after(500, self.computer_move)

    def make_move(self, board, move):
        sr, sc = move['start']
        er, ec = move['end']
        board[er][ec] = board[sr][sc]
        board[sr][sc] = ""
        if move['jump']:
            jr, jc = move['jump']
            board[jr][jc] = ""

    def computer_move(self):
        moves = self.get_all_moves(self.board, 'c')
        if not moves:
            self.end_game("У комп'ютера немає ходів. Ви перемогли! 🎉")
            return

        best_score = -float('inf')
        best_move = None

        for move in moves:
            b_copy = copy.deepcopy(self.board)
            self.make_move(b_copy, move)
            if move['end'][0] == SIZE - 1 and b_copy[move['end'][0]][move['end'][1]] == 'c':
                b_copy[move['end'][0]][move['end'][1]] = 'C'

            score = self.minimax(b_copy, 3, False, -float('inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            self.make_move(self.board, best_move)
            er, ec = best_move['end']
            if er == SIZE - 1 and self.board[er][ec] == 'c':
                self.board[er][ec] = 'C'

            self.update_ui()

            if not self.get_all_moves(self.board, 'p'):
                self.end_game("Комп'ютер переміг! 🤖")
                return

        self.turn = "p"
        self.label.config(text="Ваш хід! Оберіть шашку")

    def minimax(self, board, depth, maximizing, alpha, beta):
        if depth == 0:
            return self.evaluate_board(board)

        player = 'c' if maximizing else 'p'
        moves = self.get_all_moves(board, player)

        if not moves:
            return -1000 if maximizing else 1000

        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                b_copy = copy.deepcopy(board)
                self.make_move(b_copy, move)
                eval = self.minimax(b_copy, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                b_copy = copy.deepcopy(board)
                self.make_move(b_copy, move)
                eval = self.minimax(b_copy, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

    def evaluate_board(self, board):
        score = 0
        for r in range(SIZE):
            for c in range(SIZE):
                val = board[r][c]
                if val == 'c': score += 10
                elif val == 'C': score += 35
                elif val == 'p': score -= 10
                elif val == 'P': score -= 35
        return score

    def end_game(self, message):
        self.label.config(text=message)
        self.turn = "GameOver"

    def reset_game(self):
        self.board = [
            ["", "c", "", "c", "", "c"],
            ["c", "", "c", "", "c", ""],
            ["", "", "", "", "", ""],
            ["", "", "", "", "", ""],
            ["", "p", "", "p", "", "p"],
            ["p", "", "p", "", "p", ""]
        ]
        self.turn = "p"
        self.selected_piece = None
        self.valid_moves = []
        self.active_jump_piece = None
        self.label.config(text="Ваш хід! Оберіть шашку")
        self.update_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = CheckersProGame(root)
    root.mainloop()