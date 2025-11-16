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
        self.C = 1.4
        self.T=4000
        self.time_limit_per_movement= 10
        self.simulation_depth_limit=42


    def mount(self, T = None, C=None, time_limit_per_movement=None, simulation_depth_limit=None):
        if C is not None:
            self.C = C
        if T is not None:
            self.T=T
        if time_limit_per_movement is not None:
            self.time_limit_per_movement= time_limit_per_movement
        if simulation_depth_limit is not None:
            self.simulation_depth_limit= simulation_depth_limit

    def act(self, s: np.ndarray) -> int:
        
        num_1 = np.sum(s == 1) 
        num_m1 = np.sum(s == -1)

        if num_1 > num_m1:
            player = -1
        else:
            player = 1

        state = ConnectState(s.copy(), player)

        if state.get_winner() is not None:
            return state.get_free_cols()[0]

        player = state.player
        root = self.Node(state, None, None)
        
        time_limit=self.time_limit_per_movement
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            node = root
            
            #Selección
            while( node.state.get_winner() is None and node.children != {}): # Hasta que el juego acabe o hayan ramas por explorar
                node = self.select_ucb(node)
            
            #Expansión
            if (node.state.get_winner() is None and node.candidates_actions != []):
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
            next_state_other= ConnectState(node.state.board.copy(),other_player)
            next_step_other= next_state_other.transition(action)

            if next_step_other.get_winner()==other_player:
                node.candidates_actions.remove(action)
                no_play_state=node.state.transition(action)
                child=self.Node(no_play_state,node,action)

                node.children[action]=child

                return child

        if node.candidates_actions:
            action=node.candidates_actions.pop()
            next_state=node.state.transition(action )
            child= self.Node(next_state,node,action)
            node.children[action]=child

            return child
    
    def innerTrial(self, state: ConnectState, player:int):

        'Primero intenta ganar, si no puede entonces bloquea al oponente. Si no hay riesgo claro juega aleatorio'

        depth=0


        while True:
            
            winner = state.get_winner()

            if winner is not None:
                if winner == player:
                    return 1
                elif winner == 0:
                    return 0
                
                return -1
            
            if depth>=self.simulation_depth_limit:
                return 0

            play=False
            legal_actions= state.get_free_cols()
            
            for action in legal_actions:
                next_state= state.transition(action)

                if next_state.get_winner()== state.player:
                    state=next_state
                    play=True
                    break
            
            if play== True:
                depth=depth+1
                continue
            
            #Puede ganar el enemigo
            other=-state.player

            for act in legal_actions:

                other_player_state= ConnectState(state.board.copy(),other)
                next_other_player_state = other_player_state.transition(act)

                if next_other_player_state.get_winner()==other:
                    state=state.transition(act)
                    play=True
                    break
                
            if play:
                depth=depth+1
                continue


            action = random.choice(legal_actions)
            state = state.transition(action)
            depth=depth+1
        
    
    def propagation(self, node, R: float):
        
        while node is not None:
            node.N += 1
            node.R += R
            R = -R # Porque es de juego de suma cero
            node = node.parent
    
    
    def takeAction(self, node):

        best_q_value=-float("inf")
        best_action = None

        for action, child in node.children.items():
            if child.N > 0:
                q= child.R/child.N
            else:
                q=0

            if q> best_q_value:
                best_q_value=q
                best_action = action
        
        if best_action is None:
            if node.children:
                visits = -1
                
                for action, child in node.children.items():
                    if child.N > visits:
                        visits=child.N
                        best_action=action
        
        if best_action is None:
            best_action= node.state.get_free_cols()[0]
        

        return best_action

    
    