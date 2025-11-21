import math
import random
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
import time
       
class MCTS(Policy):
    
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
        self.T = 42
        self.C = 1.4

    def mount(self, T: int=42, C: float = 1.4):
        self.T = T
        self.C = C


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
            player=1
        else:
            player=-1

        state = ConnectState(s.copy(), player)

        if state.is_final():
            return 0

        if not state.get_free_cols():
            return 0

        for cols in state.get_free_cols():
            next_step_copy=state.transition(cols)
            if next_step_copy.get_winner()==player:
                return cols

        other= -player

        for cols in state.get_free_cols():
            state_other=ConnectState(state.board.copy(), other).transition(cols)

            if state_other.get_winner()==other:
                return cols    


        problems= self.problem_two_movements(state, player)
        
        good_movement=[]

        for column in state.get_free_cols():
            if column not in problems:
                good_movement.append(column)

        if good_movement:
            allowed_movements=good_movement
        else:
            allowed_movements=state.get_free_cols()


        
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
        
        return self.takeAction(root)
            

    
    def select_ucb(self, node):
        
        succ = None     # El succesor escogido, no la lista
        current_ucb = -float("inf")
        
        for action, child in node.children.items():
            
            if child.N > 0:
                Q = child.R/child.N
            else:
                Q = 0
            
            exploration_factor =  self.C * math.sqrt( math.log(node.N + 1) / (child.N + 1))  
            
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


            if not state.get_free_cols():
                return 0
            
            for action in state.get_free_cols():
                    
                    next_step_copy=state.transition(action)

                    if next_step_copy.get_winner()== state.player:
                        play=action
                        break
            
            if play is None:
                other=-state.player

                for act in state.get_free_cols():

                    other_player_state= ConnectState(state.board.copy(),other)
                    next_other_player_state = other_player_state.transition(act)

                    if next_other_player_state.get_winner()==other:
                        play = act
                        break
                
            if play is None:
                play = random.choice(state.get_free_cols())
            
            
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

        if not node.state.get_free_cols():
            return 0

        for action, child in node.children.items():
            if action not in node.state.get_free_cols():
                continue

            if child.N > 0:
                q= child.R/child.N
            else:
                q=0

            if q> best_q_value:
                if action in node.state.get_free_cols():
                    best_q_value=q
                    best_action = action
        
        if best_action is None:
            if node.children:
                visits = -1
                
                for action, child in node.children.items():

                    if action not in node.state.get_free_cols():
                        continue
                    if child.N > visits:
                        visits=child.N
                        best_action=action
        
        if best_action is None:
            best_action= random.choice(node.state.get_free_cols())
        

        return best_action
    
    def problem_two_movements(self, state: ConnectState, player:int):

        other_player=-player
        future_problems=[]


        for col in state.get_free_cols():

            new_state = state.transition(col)

            problems_count=0

            for one in new_state.get_free_cols():
                other_state = new_state.transition(one)

                if other_state.get_winner() == other_player:

                    problems_count= problems_count+1
                
                if problems_count >=2:
                    break

            if problems_count >= 2:
                future_problems.append(col)

        return future_problems

