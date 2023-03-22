

# Importamos librerias.

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
# Direcciones de movimiento.
class Direccion(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Coordenadas ( x , y ) -> Punto.
    
Punto = namedtuple('Punto', 'x, y')

# Colores -  rgb. 
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20

# Construimos nuestra clases V-IA.

class ViboritaInteligente:

	# Constructor.
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Viborita Inteligente.')
        self.clock = pygame.time.Clock()
        
        # init game state - resetear el juego.

    def reset(self):
		
        self.direction = Direccion.RIGHT
        
        self.head = Punto(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Punto(self.head.x-BLOCK_SIZE, self.head.y),
                      Punto(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration=0
        # Premio.
        self.reward=0

        # Posicion de la comida.
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Punto(x, y)
        if self.food in self.snake:
            self._place_food()

            # Funcion jugar a un pasp.
        
    def play_step(self,action):
	 self.frame_iteration+=1
        # 1. collect user input
           for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. movimiento.
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. checar si perdiò.
        self.reward=0
        game_over = False
        if self._is_collision()or self.frame_iteration>100*len(self.snake):
            game_over = True
            # Premio +- 10.
            self.reward = -10 
            return  self.reward ,game_over, self.score
            
        # 4. Nuevo lugar de comida - muevete.
        if self.head == self.food:
            self.score += 1
            # Premio de comida +10
            self.reward +=10
            self._place_food()
        else:
            self.snake.pop()

            # Penalizar.
        if self.frame_iteration > 20:

		self.reward -= int(0.001*self.frame_iteration)
      
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Perdiste, regresa el juego y record a 0.
        return game_over, self.score

# Funcion importantisima - Colisiones
    
    def _is_collision(self, pt = None):

	if pt is None:
		pt=self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False

# Cuadro de pantalla.
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

        # Funcion moverse.
        
    def _move(self, action):

	clock_wise = [ Direccion.RIGHT,Direccion.DOWN,Direccion.LEFT,Direccion.UP]
	idx = clock_wise.index(self.direction)

	if np.array_equal(action,[1,0,0]):
		new_dir = clock_wise[ idx] # no cambia.
	elif np.array_equal(action,[ 0,1,0]):
		next_idx = (idx +1)%4
		# Direccion r -> d -> l -> u 
		new_dir = clock_wise[ next_idx]
	else: #[ 0,0,1]

		next_idx = (idx-1)%4
		# Direccion r -> u -> l -> d
		new_dir= clock_wise[ next_idx]

	self.direction = new.dir

	x = self.head.x
	y = self.head.y

	if direction == Direccion.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direccion.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direccion.DOWN:
            y += BLOCK_SIZE
        elif direction == Direccion.UP:
            y -= BLOCK_SIZE
            
        self.head = Punto(x, y)
            

#if __name__ == '__main__':
 #   game = SnakeGame()
    
    # game loop
  #  while True:
   #     game_over, score = game.play_step()
        
       # if game_over == True:
        #    break
        
    #print('Final Score', score)
        
        
    #pygame.quit()
