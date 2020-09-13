import pygame
import random as rand
from math import sqrt
import neat
import os
import pickle
import sys

ww, wh = 800, 600
#font = pygame.SysFont("Arial")

class Entity:

	def __init__(self, x, y, w, h):
		self.x = x
		self.y = y
		self.w = w
		self.h = h

	def collides(self, other):
		if (self.x <= other.x and self.x+self.w >= other.x) or (self.x<=other.x+other.w and self.x+self.w>=other.x):
			if (self.y <= other.y and self.y+self.h >= other.y) or (self.y<=other.y+other.h and self.y+self.h>=other.y):
				return True
		return False

class Paddle(Entity):
	WIDTH = 96
	HEIGHT = 24
	SPEED = 8

	def __init__(self, x, y):
		super().__init__(x,y,self.WIDTH,self.HEIGHT)
		self.dx = 0

	def update(self):
		self.x += self.dx
		if self.x < 0:
			self.x = 0
		elif self.x+self.w > ww:
			self.x = ww-self.w

	def render(self, surf):
		pygame.draw.rect(surf, (255,255,255), (int(self.x), self.y, self.WIDTH, self.HEIGHT))

class Ball(Entity):
	r = 8 # radius
	SPEED = 10/sqrt(2) # this way it has an absolute velocity of 10, split into the 2 dimensions (x,y)

	def __init__(self, x=-1, y=-1):
		super().__init__(x if x>=0 else ww/2-self.r/2, y if y >= 0 else wh*3/4,self.r,self.r)
		self.dx = (1 if rand.randint(1,10) > 5 else -1)*self.SPEED
		self.dy = -self.SPEED

	def update(self):
		self.x += self.dx
		self.y += self.dy
		if self.x-self.r <= 0 or self.x+self.r >= ww: # if the left or right side of the screen has been reached
			self.dx *= -1
		if self.y-self.r <= 0: # if the ball hit the top of the screen
			self.dy = self.r
		elif self.y+self.r >= wh: # if the ball fell to the bottom of the screen
			self.x = ww/2-self.r/2
			self.y = wh*3/4
			self.dx = (1 if rand.randint(1,10) > 5 else -1)*self.SPEED
			self.dy = -self.SPEED
			return True
		return False

	def render(self, surf):
		pygame.draw.circle(surf, (255,255,255), (int(self.x), int(self.y)), self.r, self.r)

class Obstacle(Entity):

	def __init__(self, x, y, w=64, h=24):
		super().__init__(x,y,w,h)

	def render(self, surf):
		pygame.draw.rect(surf, (255,255,255), (self.x, self.y, self.w, self.h))

def main(genomes, config):

	obstacles_amt = 60 # maximum 10 per row

	nets = []
	paddles = []
	balls = []
	obstacles = []
	ge = []

	for _, g in genomes:
		nets.append(neat.nn.FeedForwardNetwork.create(g, config)) # create actual neural network
		paddles.append(Paddle(ww/2-Paddle.WIDTH/2, wh*5/6)) # create paddle
		g.fitness = 0
		ge.append(g) # append genome
		game_obstacles = [] # generate game's obstacles
		for j in range(int(obstacles_amt/10)):
			game_obstacles += [Obstacle(60+i*69, 36*(j+2)) for i in range(int(10))] # nice
		obstacles.append(game_obstacles) # and store them
		balls.append(Ball()) # generate ball
	print("lengths:"+"\nnets: " + str(len(nets)) + "\npaddles: "+str(len(paddles))+"\nballs: "+str(len(balls))+"\nobstacles: "+str(len(obstacles))+"\nge: "+str(len(ge)))

	win = pygame.display.set_mode((ww,wh))
	surf = pygame.Surface((ww,wh))
#	clock = pygame.time.Clock()
#	tps = 30
	running = True

	while running:

#		clock.tick(tps)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		surf.fill((0,0,0))
		to_rem = []
		for i in range(len(paddles)):
			# inputs -> paddle x, ball x, ball y
			output = nets[i].activate((paddles[i].x, balls[i].x, balls[i].y))[0]

			if output >= 1/3:
				paddles[i].dx = Paddle.SPEED
			elif output <= -1/3:
				paddles[i].dx = -Paddle.SPEED
			else:
				paddles[i].dx *= 0.85

			paddles[i].update()

			if balls[i].update(): # if the ball reached the bottom
				ge[i].fitness -= 5
				to_rem.append((ge[i],balls[i],obstacles[i],paddles[i],nets[i]))
			if balls[i].collides(paddles[i]): # if the ball hit the paddle
				balls[i].dy *= -1
				ge[i].fitness += 2

			paddles[i].render(surf)
			balls[i].render(surf)
			rem = []
			for obstacle in obstacles[i]:
				obstacle.render(surf)
				if balls[i].collides(obstacle):
					rem.append(obstacle)
					ge[i].fitness += 1
					balls[i].dy *= -1
			for ob in rem:
				obstacles[i].remove(ob)
				if len(obstacles[i]) == 0:
					print("Genome #" + str(i) + " finished the game!")
					running = False

			win.blit(surf, (0,0))
			pygame.display.flip()

		if len(paddles) == 0:
			running = False

		for i in to_rem:
			ge.remove(i[0])
			balls.remove(i[1])
			obstacles.remove(i[2])
			paddles.remove(i[3])
			nets.remove(i[4])

def run(config_file):
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	winners = []

	winner = p.run(main, 1)
	winners.append(winner)
	while winner.fitness < 45:
		winner = p.run(main, 1)
		winners.append(winner)

	visualize = input("Do you want to visualize every generation? y/n: ").lower()
	visualize = True if visualize[0] == 'y' else False

	if visualize:
		for genome in winners:
			main([(1,genome)], config)

#	with open('winner-ctrnn', 'wb') as f:
#		pickle.dump(winner, f)

def run_best(config_path):
	with open('winner-ctrnn', 'rb') as f:
		genome = pickle.load(f)

	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						neat.DefaultSpeciesSet, neat.DefaultStagnation,
						config_path)
	genomes = [(1,genome)]
	main(genomes, config)

if __name__ == '__main__':
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-feedforward.txt')
	if len(sys.argv) > 1:
		if sys.argv[1] == 'best':
			run_best(config_path)
		else:
			run(config_path)
	else:
		run(config_path)