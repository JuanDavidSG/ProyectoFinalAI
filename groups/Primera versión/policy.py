import math
import random
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
       
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
        self.T = 2000
        self.C = 1.4

    def mount(self, T: int=1000, C: float = 1.4):
        self.T = T
        self.C = C


    def act(self, s: np.ndarray) -> int:
        
        num_1 = np.sum(s == 1)
        num_m1 = np.sum(s == -1)

        if num_1 == num_m1:
            player = -1 
        else:
            player = 1

        s = ConnectState(s.copy(), player)
        
        player = s.player
        root = self.Node(s, None, None)
        
        for i in range(self.T):
            node = root
            
            #Selección
            while(not node.state.is_final() and (len(node.children) == len(node.candidates_actions))): # Hasta que el juego acabe o se prueben todas las opciones
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
        
        while not state.is_final():
            
            action = random.choice(state.get_free_cols())
            state = state.transition(action)
            
        winner = state.get_winner()
        
        if winner == player:
            return 1
        elif winner == 0:
            return 0
        
        return -1
    
    
    def propagation(self, node, R: float):
        
        while node is not None:
            node.N += 1
            node.R += R
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

    
    