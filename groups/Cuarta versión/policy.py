import math
import random
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class MCTS(Policy):
    last_instance = None

    class Node():
        def __init__(self, state, parent, action):
            self.state = state
            self.parent = parent 
            self.action = action          
            self.children = {}            
            self.N = 0                    
            self.R = 0                
            self.candidates_actions = list(state.get_free_cols())

    def __init__(self):
        self.T = 35
        self.C = 1.4
        self.nodes_explored = []
        MCTS.last_instance = self

    def mount(self, T: int=42, C: float = 1.4):
        self.T = T 
        self.C = C

    def dynamiC(self, state):
        total_pieces = 0

        for col in state.board:
            for cell in col:
                if cell != 0:
                    total_pieces += 1

        progress = total_pieces / 42.0

        if progress < 0.3: 
            return self.C * 1.3  
        elif progress < 0.7:  
            return self.C  
        else:  
            return self.C * 0.8 


    def act(self, s: np.ndarray) -> int:
        
        player1_count=0
        player2_count=0

        for col in s:
            for count in col:
                if count ==1:
                    player1_count=player1_count+1
                
                elif count==-1:
                    player2_count=player2_count+1

            
        if player1_count == player2_count:
            player=-1
        else:
            player=1

        state = ConnectState(s.copy(), player)

        if state.is_final():
            self.nodes_explored.append(0)
            return 0

        # traer alturas solo una vez
        free_cols = state.get_free_cols()
        
        if not free_cols:
            self.nodes_explored.append(0)
            return 0

        # Tratar de ganar
        for cols in free_cols:
            next_step_copy=state.transition(cols)
            if next_step_copy.get_winner()==player:
                self.nodes_explored.append(1)
                return cols

        # Bloquear victoria del otro
        other= -player
        for cols in free_cols:
            state_other=ConnectState(state.board, other).transition(cols)
            
            if state_other.get_winner()==other:
                self.nodes_explored.append(1)
                return cols    


        problems= self.problem_two_movements(state, player)
        
        actions_score = []
        for col in free_cols:
            score = 0.0
            score += self.center_score(col)
            score += self.player_strategy(state, col, player)
            
            # contar mis amenazas
            my_threats = 0
            next_state = state.transition(col)
            for next_col in next_state.get_free_cols():
                if next_state.transition(next_col).get_winner() == player:
                    my_threats += 1
            score += my_threats
            
            # contar amenazas del oponente  
            opp_threats = 0
            for next_col in next_state.get_free_cols():
                if next_state.transition(next_col).get_winner() == -player:
                    opp_threats += 1
            score -= opp_threats
            
            if col in problems:
                score -= 2
                
            
            
            actions_score.append((score, col))
        
        actions_score.sort(reverse=True)

        if len(actions_score) >= 5:
            x = 5
        else:
            x = len(actions_score) 
        

        allowed_movements = [] 

        i = 0
        while i < x:
            actionS = actions_score[i]
            col = actionS[1]
            
            allowed_movements.append(col)
            i += 1

        if not allowed_movements:
            allowed_movements = free_cols.copy()
        else:
            # O todas son iguales
            random.shuffle(allowed_movements)
        
        root = self.Node(state, None, None)
        root.candidates_actions= allowed_movements.copy()
        
        for i in range(self.T):

            node = root
            
            #Selección
            while( node.state.get_winner() ==0 and node.children != {}): # Hasta que el juego acabe o hayan ramas por explorar
                node = self.select_ucb(node)
            
            #Expansión
            if (node.state.get_winner() == 0 and node.candidates_actions != []):
                node = self.expand(node)
            
            #Simulación
            R = self.innerTrial(node.state, player)
            
            #Backpropagation
            self.propagation(node, R)

            if i > 10:
                mejor_q = -1

                for action, child in root.children.items():
                    if child.N > 0:
                        q = child.R / child.N
                        if q > mejor_q:
                            mejor_q = q

                if mejor_q > 0.7:
                    break

        self.nodes_explored.append(root.N)

        return self.takeAction(root)
            

    
    def select_ucb(self, node):
        # Primero seleccionar todos los no visitados para que sirva el UCB
        unvisited = []
        for action, child in node.children.items():    
            if child.N == 0:
                unvisited.append(child)
        
        if unvisited:
            return random.choice(unvisited)
                
        succ = None
        current_ucb = -float("inf")

        dynamic_C = self.dynamiC(node.state)
        
        for action, child in node.children.items():
            
            if child.N > 0:
                Q = child.R/child.N
            else:
                Q = 0
            
            exploration_factor =  dynamic_C * math.sqrt( math.log(node.N + 1) / (child.N + 1) )   
            
            new_ucb = Q + exploration_factor  
            
            if new_ucb > current_ucb:
                current_ucb = new_ucb
                succ = child
        
        return succ                    
    
      
    def expand(self, node):
        
        legal_action= node.candidates_actions.copy()
        reorder_actions=random.sample(legal_action, k=len(legal_action))

        for action in reorder_actions:
            next_state = node.state.transition(action)

            if next_state.get_winner()== node.state.player:
                node.candidates_actions.remove(action)
                child = self.Node(next_state, node, action)
                child.candidates_actions = list(child.state.get_free_cols())
                node.children[action] = child
            
                return child
            
        other_player= -node.state.player

        for action in reorder_actions:
            next_state_other= ConnectState(node.state.board.copy(),other_player).transition(action)

            if next_state_other.get_winner()==other_player:

                node.candidates_actions.remove(action)
                no_play_state=node.state.transition(action)
                child=self.Node(no_play_state,node,action)
                child.candidates_actions=no_play_state.get_free_cols()
                node.children[action] =child

                return child

        if node.candidates_actions:
            action=node.candidates_actions.pop()
            next_statee=node.state.transition(action )
            child= self.Node(next_statee,node,action)
            child.candidates_actions = list(child.state.get_free_cols())
            node.children[action]=child

            return child
    
    def innerTrial(self, state: ConnectState, player:int):
        
        'Primero intenta ganar, si no puede entonces bloquea al oponente. Si no hay riesgo claro juega aleatorio'

        while True:
            
            winner = state.get_winner()

            if winner !=0:
                if winner == player:
                    return 1
                else:
                    return -1
            

            play=None


            free_cols = list(state.get_free_cols())
            
            if not free_cols:
                return 0
            
            # intentar ganar
            for action in free_cols:
                    
                    next_step_copy=state.transition(action)

                    if next_step_copy.get_winner()== state.player:
                        play=action
                        break
            
            # bloquear si es necesario
            if play is None:
                other=-state.player

                for act in free_cols:

                    other_player_state= ConnectState(state.board,other)
                    next_other_player_state = other_player_state.transition(act)

                    if next_other_player_state.get_winner()==other:
                        play = act
                        break
                
            if play is None:
                play = random.choice(free_cols)
            
            
            state = state.transition(play)
        
    
    def propagation(self, node, R: float):
        
        while node is not None:
            node.N += 1
            node.R += R
            R = -R # Porque es de juego de suma cero
            node = node.parent
    
    
    def takeAction(self, node):

        best_q_value=-float("inf")
        best_action = None
        eps = 1e-6

        if not node.state.get_free_cols():
            return 0

        for action, child in node.children.items():
            if action not in node.state.get_free_cols():
                continue

            if child.N > 0:
                q= child.R/child.N
            else:
                q=0

            q_ajus = max(q, 0.0)
            score = (q_ajus + eps) * (child.N + 1)

            if score > best_q_value:
                if action in node.state.get_free_cols():
                    best_q_value=score
                    best_action = action
        
        if best_action is None:
            if node.children:
                visits = float("inf")
                
                for action, child in node.children.items():

                    if action not in node.state.get_free_cols():
                        continue
                    if child.N < visits:
                        visits=child.N
                        best_action=action
        
        if best_action is None:
            best_action= random.choice(node.state.get_free_cols())
        

        return best_action
    
    def problem_two_movements(self, state: ConnectState, player:int):

        other_player=-player
        future_problems=[]

        free_cols = state.get_free_cols()
        for col in free_cols:

            new_state = state.transition(col)

            problems_count=0

            for one in new_state.get_free_cols():
                other_state=new_state.transition(one)

                if other_state.get_winner() == other_player:

                    problems_count= problems_count+1
                
                if problems_count >=2:
                    future_problems.append(col)
                    break

        return future_problems

    
    def center_score(self, col):
        center = 3
        distance = abs(col-center)
        return -distance
    
    # si es -1 prefiere impares, si es 1 al contrario
    def player_strategy(self, state, col, player):
        
        h = 0
        for row in range(state.ROWS):
            if state.board[(state.ROWS -1) - row, col] == 0:
                h = row+1
                break
        
        if h == 0:
            return -1
        
        # Condiciones jugador 1
        if player == -1:
            if h % 2 == 1:
                return 1
            else:
                return 0
        
        # Condiciones jugador dos
        if h % 2 == 0:
            return 1
        
        return 0