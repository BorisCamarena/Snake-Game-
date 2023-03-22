
# Aqui importamos las librerias previamente instaladas en la consola.
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direccion, Punto
from model import Linear_QNet, QTrainer
from helper import plot
from grafica import graficar

# Linea importante en el codigo.
MAX_MEMORIA = 100_000
# Definimos el tamañano de la cola.

T_COLA = 1000
LR = 0.001

# Creamos la clase agente.

class Agente:

	# Aquì se tiene el constructor:
	# modelo - red neuronal.
	# Optimizador. Mas adelante veremos lo curioso en el gradiente.

    def __init__(self):
	    # Se omiten algunas traducciones, por que son inmediatas.
        self.n_games = 0
        self.epsilon = 0 # juegos al azar.
        self.gamma = 0.9 # tasa de descuento.
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # Parte importante en el còdigo, por que
        # se crea una pila con el popleft().
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

# Creamos nuestra Clase estado - Agente.

    def ob_estado(self, juego):
	    # definimos la cabeza de la vibora.
        head = juego.snake[0]

        # Definimos el tamaño del pixel 20x20
        
        point_l = Punto(head.x - 20, head.y)
        point_r = Punto(head.x + 20, head.y)
        point_u = Punto(head.x, head.y - 20)
        point_d = Punto(head.x, head.y + 20)

        # Definimos las direcciones, que es super
        # importante para definir el comportamiento
        # de la viborita.
        
        dir_l = juego.direction == Direccion.LEFT
        dir_r = juego.direction == Direccion.RIGHT
        dir_u = juego.direction == Direccion.UP
        dir_d = juego.direction == Direccion.DOWN

        state = [

		# Importancia de definir las direcciones.
		
            # Peligro enfrente.
            (dir_r and juego.is_collision(point_r)) or 
            (dir_l and juego.is_collision(point_l)) or 
            (dir_u and juego.is_collision(point_u)) or 
            (dir_d and juego.is_collision(point_d)),

            # Peligro a la derecha.
            (dir_u and juego.is_collision(point_r)) or 
            (dir_d and juego.is_collision(point_l)) or 
            (dir_l and juego.is_collision(point_u)) or 
            (dir_r and juego.is_collision(point_d)),

            # Peligro a la inzquierda.
	    
            (dir_d and juego.is_collision(point_r)) or 
            (dir_u and juego.is_collision(point_l)) or 
            (dir_r and juego.is_collision(point_u)) or 
            (dir_l and juego.is_collision(point_d)),

	    # IMPORTANTE - Direccion de movimiento.
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Comida - posicion.

            juego.food.x < juego.head.x,  # comida(izquierda)
            juego.food.x > juego.head.x,  # comida(derecha)
            juego.food.y < juego.head.y,  # comida(arriba)
            juego.food.y > juego.head.y  # comida(abajo)
            ]

        # Aqui lo que hace es estar en enteros ( 0 ò 1 ).

        return np.array(state, dtype=int)

# Introducimos el estado, accion, premio, sig_estado, echo.

    def recordar(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached - si alcanza memoria maxima.

# Entrenamiento de memoria, se trabajara con tuplas ( lista de tuplas ).

    def entrenar_memoria_larga(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, T_COLA) 
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

 # Introducimos la funcion analoga - memoria a corto plazo.
 # Entrenamiento.

    def entrenar_memoria_corta(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

# Funcion super importante. Se deciden acciones.
# Con enfoque probabilistico ?

    def ob_accion(self, state):
        # Movimientos al azar ( exploracion - explotacion ).
        self.epsilon = 80 - self.n_games
        # Movimiento final.
        final_move = [0,0,0]
        # Vamos a generar ennteros 0 - 2.
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            # Dado el estado (R11) genera predict(R3)
            prediction = self.model(state0)
            # Como se definio en el if, move genra enteros entre 0 - 2
            # que es max_predict(R3)
            move = torch.argmax(prediction).item()
            # Movimiento final.
            final_move[move] = 1

            # Decision -> ( vector ) perteneciente a R3 con 0 - 1.

        return final_move

# Creamos la funcion entreamiento.

def entrenar():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agente = Agente()
    juego = ViboritaInteligente()
    while True:
        # estado viejo.
        state_old = agente.ob_estado(juego)

        # obtener movimiento.
        final_move = agente.ob_accion(state_old)

        # Mover -> nuevo estado.
        reward, done, score = juego.play_step(final_move)
        state_new = agente.ob_estado(juego)

        # entrenar memoria corta.
        agente.entrenar_memoria_corta(state_old, final_move, reward, state_new, done)

        # recordar
        agente.recordar(state_old, final_move, reward, state_new, done)

        if done:
            # Graficamos el resultado con el entrenamiento
            # de la memoria larga.
            juego.reset()
            agente.n_games += 1
            agente.entrenar_memoria_larga()

            if score > record:
                record = score
                agente.model.save()

            print('Juego', agente.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agente.n_games
            plot_mean_scores.append(mean_score)
            graficar(plot_scores, plot_mean_scores)

# Funcion principal.

if __name__ == '__main__':
    entrenar()
