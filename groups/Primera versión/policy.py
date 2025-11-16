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
            self.candidates_actions = state.get_free_cols()

    def __init__(self):
        self.C = 1.4
        self.T=4000
        self.time_limit_per_movement= 0.6
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

        if num_1 == num_m1:
            player = 1 
        else:
            player = -1

        s = ConnectState(s.copy(), player)
        
        player = s.player
        root = self.Node(s, None, None)
        
        time_limit=9.0
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            node = root
            
            #Selección
            while(not node.state.is_final() and node.children != {}): # Hasta que el juego acabe o hayan ramas por explorar
                node = self.select_ucb(node)
            
            #Expansión
            if (not node.state.is_final() and node.candidates_actions != []):
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
        
        action = node.candidates_actions.pop()
        new_state = node.state.transition(action)
        child = self.Node(new_state, node, action)
        node.children[action] = child
        
        return child
    
    
    def innerTrial(self, state: ConnectState, player:int):

        'Primero intenta ganar, si no puede entonces bloquea al oponente. Si no hay riesgo claro juega aleatorio'

        depth=0
        current_player=state.player
        next_player=-current_player

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

                if next_state.get_winner()== player:
                    state=next_state
                    play=True
                    break
            
            if play== True:
                depth=depth+1
                continue
            
            for action in legal_actions:
                next_state=state.transition(action)
                if next_state.get_winner()== -state.player:
                    state=next_state
                    play=True
                    break

            if play ==True:
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
        visits = -1
        best_action = None
        for action, child in node.children.items():
            if child.N > visits:
                visits = child.N
                best_action = action
        if best_action == None:
            best_action = node.state.get_free_cols()
            best_action = best_action[0]
        
        return best_action

    
    