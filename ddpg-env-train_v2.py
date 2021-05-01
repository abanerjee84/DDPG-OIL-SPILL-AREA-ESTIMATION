from __future__ import division
import pygame
import math
import random
import pandas as pd
from pygame.locals import *
from pygame import gfxdraw
import spill as ss
import time
import numpy as np
import copy
from shapely.geometry import Point, Polygon,MultiPoint,MultiLineString
from shapely.geometry.polygon import LinearRing,orient,LineString

import os
import psutil
import gc
import train
import buffer


cwd = os.getcwd()
TWO_PI = 2 * np.pi
## SURFACE PROPERTIES ##

background_colour = (255,255,255)
(width, height) = (800, 800)

screen = pygame.display.set_mode((width, height))#,pygame.FULLSCREEN)
trajectorysurface = pygame.Surface((width, height))
trajectorysurface.fill((255,255,255))
contoursurface = pygame.Surface((width, height))
contoursurface.fill((255,255,255))
contoursurface_refined = pygame.Surface((width, height))
contoursurface_refined.fill((255,255,255))
actspill_surface = pygame.Surface((width, height))
actspill_surface.fill((255,255,255))
screen.fill(background_colour)
pygame.display.set_caption('SPILL-DDPG')
pygame.font.init()

font = pygame.font.SysFont("monospace", 24,bold=True)


# ==========================
#   Training Parameters
# ==========================
MAX_TOTAL_REWARD = 300
# Max training steps
MAX_EPISODES = 1000
# Max episode length
MAX_STEPS = 200
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# Parameters for neural net
HIDDEN1_UNITS = 128
HIDDEN2_UNITS =64
L2_REG_SCALE = 0
MAX_BUFFER = 1000000
MINIBATCH_SIZE = 500
## ENV ##
S_DIM =7
A_DIM = 6
A_MAX = 1


## DRONE PROPERTIES ##
numdrones=7
radio=150
disttospill=10
flight_time=20

## DRONE DEPLOY POINT ##
dronedeploy_x=-150
dronedeploy_y=-150

## DRONE GENERAL DIRECTION ##
drone_general_direction="ne"
gd_x,gd_y=ss.generaldirection(drone_general_direction)


## SPILL CHARECTERISTICS ##
random_seed =131
spill_radian=0.3
spill_edge=0.01
spill_points=7
spill_scale=300
predict_spill=[]
predicted_area=0
actual_area_hull=0

## UAV DYNAMICS ##
W=100
rho=1.225
R=0.5
A=0.79
omega=400
U_tip=200
blades=4
aerofoil=0.0196
rsd=0.05
SFP=0.0118
d0=0.3
correctionfactor=0.1
v0=7.2
delta=0.012
P0=(delta/8)*rho*rsd*A*(omega**3)*(R**3)
Pi=(1+correctionfactor)*((W**1.5)/(np.sqrt(2*rho*A)))

## RENDER PROPERTIES ##
demo=True
render=False
footprint_render=False
path_render=True
fast_render=False
please_plot=0
time_elapsed_tot=0
avg_speed=0



def distance(i,j):
	return math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2)

class drone():
	def __init__(self):
		random.seed(time.perf_counter())
		self.x = random.randint(dronedeploy_x-1,dronedeploy_x+1)
		self.y = random.randint(dronedeploy_y-1,dronedeploy_y+1)
		self.z = random.randint(50,50)
		self.dx = 0.001
		self.dy = 0.001
		self.dz = 1
		self.lastdx=0
		self.lastdy=0
		self.angle = random.uniform(0,2*math.pi)
		self.age=1
		self.fitness=0
		self.id=0
		self.lastspillx=1000
		self.lastspilly=1000
		self.x_old=self.x
		self.y_old=self.y
		self.footprint=30
		self.color=(np.random.randint(255),np.random.randint(255),np.random.randint(255))
		self.sensor=0.5
		self.lastsensor=0.5

		self.stage=(255,0,0)
		self.total_reward=0
		self.loss=0
		self.done=False

		self.trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

		self.locations=[]
		self.area_under_me=0


	def fp(self):
		xsensor = 36 # width of sensor in mm
		ysensor = 24 # height of sensor in mm
		focallen = 50 # focal length of lens in mm
		altitude = self.z # height in m
		xgimbal = 0 # x-axis gimbal angle
		ygimbal = 0 # y-axis gimbal angle
		# Calculate field of view
		xview = 2*math.degrees(math.atan(xsensor/(2*focallen)))
		yview = 2*math.degrees(math.atan(ysensor/(2*focallen)))
		dronetobottom=altitude*math.tan(math.radians(xgimbal-.5*xview))
		dronetotop=altitude*math.tan(math.radians(xgimbal+.5*xview))
		dronetoleft=altitude*math.tan(math.radians(ygimbal-.5*yview))
		dronetoright=altitude*math.tan(math.radians(ygimbal-.5*yview))
		ht=abs(dronetoright)
		wt=abs(dronetotop)
		return ht,wt

	def quad(self):
		xy=[1,1]
		if (self.angle>0 and self.angle<1.57):
			xy=[-1,1]
		if (self.angle>1.57 and self.angle<3.14):
			xy=[-1,-1]
		if (self.angle<-3.14 and self.angle<-1.57):
			xy=[1,-1]
		if (self.angle<-1.57 and self.angle<0):
			xy=[1,1]
		return xy


	def display(self):
		# time.sleep(0.0001)
		global predicted_area
		global actual_area_hull
		ht,wt=self.fp()
		self.footprint=int(max(ht,wt))

		if self.sensor==1 and time_elapsed_tot>0.5:
			global please_plot
			please_plot=1
			global predict_spill
			predict_spill.append((int(self.x + width/2), int(self.y + height/2)))
			self.locations.append((int(self.x + width/2), int(self.y + height/2)))

			## AREA CALCULATION##
			if len(predict_spill)>3:
				poly2=Polygon(predict_spill)
				p=poly2.convex_hull
				predicted_area=p.area/1e4
				np.clip(predicted_area,0,15)

				poly2=Polygon(spill)
				p=poly2.convex_hull
				actual_area_hull=p.area/1e4
				np.clip(actual_area_hull,0,15)

			if len(self.locations)>3:
				poly2=Polygon(self.locations)
				p=poly2.convex_hull
				self.area_under_me=p.area/1e4
				np.clip(self.area_under_me,0,15)


		if (path_render==True ):
			pygame.draw.circle(trajectorysurface, (0,0,0), [int(dronedeploy_x + width/2), int(dronedeploy_y + height/2)], 20,2)
			# pygame.gfxdraw.circle(trajectorysurface,int(self.x + width/2), int(self.y + height/2),self.color)
			pygame.draw.circle(trajectorysurface, self.color, [int(self.x + width/2), int(self.y + height/2)], 1)
			screen.blit(trajectorysurface,(0,0))

			if please_plot==1 and fast_render==False:
				pygame.draw.circle(contoursurface, (0,0,0), [int(self.x + width/2), int(self.y + height/2)], 1)
				cs=pygame.transform.smoothscale(contoursurface, (int(width/2), int(height/2)))
				screen.blit(cs,(width/2+150, 0))
				if len(predict_spill)>3:
					contoursurface_refined.fill(background_colour)

					for t in range(len(predict_spill)-1):
						pygame.draw.line(contoursurface_refined,(180,180,180),[int(predict_spill[t][0]), int(predict_spill[t][1])],[int(predict_spill[t+1][0]), int(predict_spill[t+1][1])],1)
						# pygame.draw.circle(contoursurface_refined, (0,0,0), [int(predict_spill[t][0]), int(predict_spill[t][1])], 6)
					for t in range(len(predict_spill)-1):
						# pygame.draw.line(contoursurface_refined,(180,180,180),[int(predict_spill[t][0]), int(predict_spill[t][1])],[int(predict_spill[t+1][0]), int(predict_spill[t+1][1])],1)
						pygame.draw.circle(contoursurface_refined, (0,0,0), [int(predict_spill[t][0]), int(predict_spill[t][1])], 6)


					cs=pygame.transform.smoothscale(contoursurface_refined, (int(width/2), int(height/2)))
					screen.blit(cs,(width/2+150, int(height/2)-135))

					pygame.gfxdraw.filled_polygon(actspill_surface, spill, (10,10,10))
					cs=pygame.transform.smoothscale(actspill_surface, (int(width/2), int(height/2)))
					screen.blit(cs,(width/2+150, int(height)-285))

					font = pygame.font.SysFont("monospace", 18,bold=True)
					text = font.render("(c) Estimated Spill", True, (000, 000, 000))
					text2 = font.render("Area(K.M^2): {0:.2f}".format(predicted_area), True, (000, 000, 000))
					screen.blit(text, (width/2+170, int(height/2)+70))
					screen.blit(text2, (width/2+170, int(height/2)+90))
					text = font.render("(b) Spill Data", True, (000, 000, 000))
					text2 = font.render("from Flight Path.", True, (000, 000, 000))
					screen.blit(text, (width/2+170, 0+220))
					screen.blit(text2, (width/2+170, 0+245))
					text = font.render("(d) Actual Spill", True, (000, 000, 000))
					text2 = font.render("Area(K.M^2): {0:.2f}".format(actual_area_hull), True, (000, 000, 000))
					screen.blit(text, (width/2+170, int(height)-80))
					screen.blit(text2, (width/2+170, int(height)-60))
					font = pygame.font.SysFont("monospace", 20,bold=True)
					text = font.render("Area Estimation Error (\u0394A)={0:.2f}".format(actual_area_hull-predicted_area), True, (150, 00, 00))
					screen.blit(text, (width/2+10, int(height)-40))

		elif (render==True):

			# pygame.gfxdraw.filled_polygon(screen, spill, (200,200,200))
			pygame.draw.circle(screen, self.stage, [int(self.x + width/2), int(self.y + height/2)], 4)
			if footprint_render==True:
				pygame.draw.circle(screen, (0,255,0), [int(self.x + width/2), int(self.y + height/2)], self.footprint,2)
			endx = self.x - 5*math.cos(self.angle)
			endy = self.y - 5*math.sin(self.angle)
			pygame.draw.line(screen, (0,0,255), [int(self.x + width/2), int(self.y + height/2)],
				[int(endx + width/2), int(endy + height/2)], 3)


	def get_state(self):
		self.x_old=self.x
		self.y_old=self.y

		bros = []
		oneinspill=[]
		oninspillflag=False
		nearby=0

		for i in drones:
			bs_j,io_j=ss.sensor(x_spill,y_spill,(i.x + width/2),(i.y + height/2))
			if io_j==True:
				oneinspill.append((i.x,i.y))
				oninspillflag=True
			dist_j=ss.spill_bound_dist(x_spill,y_spill,(self.x + width/2),(self.y + height/2))
			bros.append([i, distance(self,i), io_j, i.x, i.y, dist_j,i.fitness])
			if distance(self,i)<2*self.footprint:
				nearby+=1

		#sort the list of other anifootprintmals on the screen by distance, this is very slow
		bros.sort(key = lambda bros: bros[1])
		bs_i,io_i=ss.sensor(x_spill,y_spill,(self.x + width/2),(self.y + height/2))
		if self.sensor==0.5:
			self.lastspillx=self.x
			self.lastspilly=self.y

		dist_i=ss.spill_bound_dist(x_spill,y_spill,(self.x + width/2),(self.y + height/2))
		# disttospill=self.footprint/2
		if dist_i<disttospill and io_i==True:
			self.stage=(255,0,0)
			self.sensor=1
		elif dist_i>disttospill and io_i==True:
			self.stage=(0,255,0)
			self.sensor=0.5
		else:
			self.stage=(0,0,255)
			self.sensor=0.1

		bs_j,io_j=ss.sensor(x_spill,y_spill,(bros[1][0].x + width/2),(bros[1][0].y + height/2))
		dist_j=ss.spill_bound_dist(x_spill,y_spill,(bros[1][0].x + width/2),(bros[1][0].y + height/2))
		if dist_j<disttospill and io_j==True:
			bs=1
		elif dist_j>disttospill and io_j==True:
			bs=0.5
		else:
			bs=0.1


		current_area=0
		current_spill=[]
		for hu in range(len(bros)):
			if (bros[hu][0].sensor==1):
				current_spill.append((int(bros[hu][0].x + width/2), int(bros[hu][0].y + height/2)))
		if len(current_spill)>3:
			poly23=Polygon(current_spill)
			p23=poly23.convex_hull
			current_area=p23.area/1e4
			np.clip(current_area,0,15)

		if bros[1][1]>radio:
			d_bro=-radio
			bs=0
		else:
			d_bro=bros[1][1]
			bs=bs

		state_raw=[self.sensor,self.angle,self.dx,self.dy,self.footprint,d_bro,bs]
		state_raw_mean=np.mean(state_raw)
		state_raw_std=np.std(state_raw)
		state_norm=[]
		for ty in range(len(state_raw)):
			state_norm.append((state_raw[ty]-np.min(state_raw) )/(np.max(state_raw)-np.min(state_raw) ))
		# print(state_norm)
		return bros,current_area,state_norm
		# return bros,[self.lastspillx,self.lastspilly,self.sensor,self.lastsensor,self.lastdx,self.lastdy]


	def maddpg(self):
		bros,current_area,state=self.get_state()
		ls=state[0]
		s_t=np.float32(state)
		# s_t=np.asarray(s_t)
		a_t = self.trainer.get_exploration_action(s_t)

		pold=self.area_under_me

		## ACTIONS ##
		# if bros[1][1]<self.footprint:
		f1=a_t[0]*1
		f2=a_t[1]*1
		self.dx += (f1)*(self.x - bros[1][0].x)
		self.dy += (f2)*(self.y - bros[1][0].y)
		# # else:
		# f1=a_t[0]*1
		# f2=a_t[1]*1
		# self.dx += (f1)*(self.x - bros[1][0].x)
		# self.dy += (f2)*(self.y - bros[1][0].y)

		f5=a_t[4]*1
		xy=self.quad()
		self.dx += xy[0]*f5
		self.dy += xy[1]*f5
		self.update()

		f3=a_t[2]*1
		f4=a_t[3]*1
		self.dx += -abs(f3)*(self.x - self.lastspillx)
		self.dy += -abs(f4)*(self.y - self.lastspilly)
		self.update()


		self.dz+=a_t[5]*2
		# self.dx +=a_t[5]*  gd_x/2
		# self.dy +=a_t[5]*  gd_y/2
		self.update()

		self.display()
		pnew=self.area_under_me

		#
		# #if you are going very fast then slow down
		# if abs(self.dx) >= 1.2:
		# 	self.dx = np.sign(self.dx)*1.2
		# if abs(self.dy) >= 1.2:
		# 	self.dy = np.sign(self.dy)*1.2

		# if (ls>0.1 and bros[1][1]>radio):
		# 	self.stage=(0,0,0)
		# 	f1=a_t[0][0]*2
		# 	f2=a_t[0][1]*2
		# 	xy=self.quad()
		# 	self.dx += -xy[0]*f1*2
		# 	self.dy += -xy[1]*f2*2
		# 	# self.z=35

			# self.z=35
		# self.update()


		# f1=a_t[0][0]*np.exp((1/(bros[1][1]+0.1))**2)*ls
		# f2=a_t[0][1]*np.exp((1/(bros[1][1]+0.1))**2)*ls
		# if bros[1][1] <= radio and self.sensor!=1:
		#
		# 	if (self.sensor==0.1 and bros[1][0].sensor==0.5):
		# 		self.dx+= -f1*(self.x - bros[1][3])*2
		# 		self.dy+= -f2*(self.y - bros[1][4])*2
		#
		# 	elif (self.sensor==0.1 and bros[1][0].sensor==1):
		# 		self.dx+= -f1*(self.x - bros[1][3])
		# 		self.dy+= -f2*(self.y - bros[1][4])
		#
		# 	elif (self.sensor==0.1 and bros[1][0].sensor==0.1):
		# 		#if you are too close move away REVERSE FLOCKING
		# 		if bros[1][1] <= radio/2:
		# 			self.dx += f1*(self.x - bros[1][0].x)
		# 			self.dy += f2*(self.y - bros[1][0].y)
		# 		#if you can see some other animals move towards them   FLOCKING
		# 		for i in bros:
		# 			if i[1] <= radio:
		# 				self.dx += -f1*(self.x - i[0].x)
		# 				self.dy += -f2*(self.y - i[0].y)
		#
		# 	elif (self.sensor==0.5 and bros[1][0].sensor==0.5):
		# 		self.stage=(255,0,0)
		# 		self.dx += f1*(self.x - bros[1][0].x)
		# 		self.dy += f2*(self.y - bros[1][0].y)
		#
		# if (self.sensor==1):
		# 	self.stage=(0,0,255)
		# 	xy=self.quad()
		# 	self.dx += -xy[0]*f1
		# 	self.dy += -xy[1]*f2
		#
		# self.update()


		bros,current_area,state_next=self.get_state()
		rs=state_next[0]

		# print(state_next[3])

		# r_t=0
		# if rs==1 and ls==0.5:
		# 	r_t=1000
		# elif rs==0.5 and ls==1:
		# 	r_t=-0.5
		# elif rs==0.1 and ls==1:
		# 	r_t=-1
		# elif rs==1 and ls==0.1:
		# 	r_t=500
		# elif rs==0.1 and ls==0.1:
		# 	r_t=-np.sqrt((self.x-self.lastspillx)**2+(self.y-self.lastspilly)**2)/100
		#
		# elif rs==1 and ls==1:
		# 	r_t=100
		# elif rs==0.5 and ls==0.5:
		# 	r_t=-0.5
		# else:
		# 	r_t=0
		#
		# r_t=r_t-self.fitness/1000
		# r_t=(rs-0.45)*1+bros[1][1]/1000

		# r_t=(rs-0.5)*10-np.sqrt((self.x-self.lastspillx)**2+(self.y-self.lastspilly)**2)/10+bros[1][1]/10
		# r_t=(rs-ls)*100 + bros[1][1]/100 - np.sqrt((self.x-self.lastspillx)**2+(self.y-self.lastspilly)**2)
		# r_t=(rs-0.5)*1000- (actual_area-predicted_area)*10-np.sqrt((self.x-self.lastspillx)**2+(self.y-self.lastspilly)**2)/100-(i.fitness/100000)-(1000/(bros[1][1]+0.1))
		uniqy=(len(set(self.locations)))
		# r_t=(rs-0.2)*10+ (uniqy/100)-(i.fitness/10000)- (actual_area-predicted_area)

		r_t=(rs-0.49)+((uniqy*5)/(time_elapsed_tot+0.001)+(actual_area-predicted_area)-(i.fitness/10000))/10

		s_t_1=np.float32(state_next)


		ram.add(s_t, a_t, r_t, s_t_1)

		# s_t = s_t_1
		self.total_reward += r_t
		# print(self.total_reward)





	def update(self):

		#if you are going very fast then slow down
		if abs(self.dx) >= 1.2:
			self.dx = np.sign(self.dx)*1.2
		if abs(self.dy) >= 1.2:
			self.dy = np.sign(self.dy)*1.2

		self.lastsensor=self.sensor
		self.lastdx=self.dx
		self.lastdy=self.dy
		timestep = 1
		self.x += (self.dx*timestep)
		self.y += (self.dy*timestep)

		if (self.z>150):
			self.z=150
		if (self.z<50):
			self.z=50
		self.age-=0.0001

		# self.fitness-=(np.random.rand()/5)*math.sqrt((self.x-self.x_old)**2+(self.y-self.y_old)**2)

		#periodic boundary conditions.
		if self.x >= width/2:
			self.done=True
			self.x = random.randint(dronedeploy_x-1,dronedeploy_x+1)
			self.y = random.randint(dronedeploy_y-1,dronedeploy_y+1)
			self.z = random.randint(50,50)
		if self.x <= -width/2:
			self.done=True
			self.x = random.randint(dronedeploy_x-1,dronedeploy_x+1)
			self.y = random.randint(dronedeploy_y-1,dronedeploy_y+1)
			self.z = random.randint(50,50)
		if self.y >= height/2:
			self.done=True
			self.x = random.randint(dronedeploy_x-1,dronedeploy_x+1)
			self.y = random.randint(dronedeploy_y-1,dronedeploy_y+1)
			self.z = random.randint(50,50)
		if self.y <= -height/2:
			self.done=True
			self.x = random.randint(dronedeploy_x-1,dronedeploy_x+1)
			self.y = random.randint(dronedeploy_y-1,dronedeploy_y+1)
			self.z = random.randint(50,50)

		self.angle = math.atan2(self.dy, self.dx)

##GET SPILLSHAPE##
np.random.seed(random_seed)
x_spill,y_spill,spill,area=ss.spill_contour(spill_radian,spill_edge,spill_points,spill_scale)
poly2=Polygon(spill)
p=poly2.convex_hull
actual_area=p.area/1e4
np.clip(actual_area,0,15)


## INIT ACTOR CRITIC ##
ram = buffer.MemoryBuffer(MAX_BUFFER)
##INIT DRONE SWARM##
drones = []
for i in range(numdrones):
	drones.append(drone())

## INIT SIM ##
tot_time=0
running = True
max_speed=0
ite=0
re_episode=0
loss_episode=0
re_epi_buff_final=[]

max_reward_index=-1
max_re=-100000



## RUN ITER ##
while running:
	for event in pygame.event.get():
	    if event.type == pygame.QUIT:
	        running = False
	    elif event.type == KEYDOWN:
	        if event.key == K_ESCAPE:
	            running = False

	screen.fill(background_colour)
	pygame.gfxdraw.filled_polygon(screen, spill, (200,200,200))

	tot_dist=0
	sum_speed=0
	re=0
	loss_now=0
	re_epi_buff=[]
	sum_energy=0

	for i in drones:

		if(time_elapsed_tot<flight_time):
			oldpos=[i.x,i.y]
			i.display()
			i.maddpg()

			newpos=[i.x,i.y]
			distance_tav=10*(abs(newpos[0]-oldpos[0])+abs(newpos[1]-oldpos[1]))
			tot_dist+=distance_tav
			time_elapsed=1
			speed=distance_tav/(time_elapsed)
			sum_speed+=speed
			V=speed
			i.fitness+=P0*(1+((3*V**2)/(U_tip**2)))
			+Pi*(np.sqrt((1+V**4)/(4*v0**4))-V**2/(2*v0**2))**0.5
			+(0.5*d0*rho*rsd*A*V**3)
			sum_energy+=i.fitness
			tot_time+=time_elapsed
			re+=i.total_reward



	if(time_elapsed_tot>flight_time):

		random.seed(time.perf_counter())
		al=0
		cl=0
		re=0

		agent_rewards=[]
		agent_rewards.append(ite)
		for i in drones:
			i.loss=0
			loss_actor,loss_critic=i.trainer.optimize()
			al+=loss_actor.item()
			cl+=loss_critic.item()
			re+=i.total_reward
			agent_rewards.append(i.total_reward)
			if i.total_reward>max_re:
				max_re=i.total_reward
				max_reward_index=i
				i.trainer.save_models_best()
			i.x = random.randint(dronedeploy_x-1,dronedeploy_x+1)
			i.y = random.randint(dronedeploy_y-1,dronedeploy_y+1)
			i.z = random.randint(50,50)
			i.dx = 0
			i.dy = 0
			i.dz = 1
			i.lastdx=0.001
			i.lastdy=0.001
			i.angle = random.uniform(0,2*math.pi)
			i.age=1
			i.fitness=0
			i.x_old=i.x
			i.y_old=i.y
			i.footprint=30
			i.color=(np.random.randint(255),np.random.randint(255),np.random.randint(255))
			i.sensor=0.5
			i.lastsensor=0.5
			i.stage=(255,0,0)
			i.done=False
			i.total_reward=0
			i.locations=[]
			i.area_under_me=0


			try:
				if np.random.rand()<0.5:
					i.trainer.load_models_best()
			except:
				pass


			# i.loss=0
			i.display()
			i.update()
		predict_spill=[]
		trajectorysurface.fill(background_colour)
		contoursurface.fill(background_colour)
		font = pygame.font.SysFont("monospace", 24,bold=True)
		text = font.render("Time Elapsed (minutes): {0:.2f}".format(time_elapsed_tot), True, (100,100,100))
		screen.blit(text, (30, 670))
		text = font.render("Spill Area (K.M^2): {0:.2f}".format(actual_area), True, (100,100,100))
		screen.blit(text, (30, 640))
		text = font.render("Avg UAV Velocity (m/sec):): {0:.2f}".format(avg_speed), True, (100,100,100))
		screen.blit(text, (30, 610))
		text = font.render("(a). Flight Path of {} UAVs".format(numdrones), True, (000, 000, 000))
		screen.blit(text, (30, 700))
		pygame.draw.line(screen, (0,0,0), [50, 575],[150,575], 3)
		pygame.draw.line(screen, (0,0,0), [50, 565],[50,585], 3)
		pygame.draw.line(screen, (0,0,0), [150, 565],[150,585], 3)
		text = font.render("100px = 1K.m", True, (200, 000, 000))
		screen.blit(text, (30, 540))
		pygame.display.flip()
		ite+=1
		if ite%10==0:
			pygame.image.save(screen, "./Train_Shots/screenshot_"+str(ite)+".png")
		re_epi_buff.append(ite)
		re_epi_buff.append(re/numdrones)
		re_epi_buff.append(al/numdrones)
		re_epi_buff.append(cl/numdrones)
		re_epi_buff_final.append(re_epi_buff)
		print("Episode Reward (Avg):",re/numdrones,al/numdrones,cl/numdrones,ite)


		df=pd.DataFrame(re_epi_buff_final,columns=['Episode','Avg Reward',"Actor Loss","Critic Loss"])
		df.to_csv(cwd+'/Plots/Maddpg_train_plot.csv')

		df=pd.DataFrame(agent_rewards)
		df.to_csv(cwd+'/Plots/Maddpg_train_plot_agents.csv')
		re_episode=0
		re=0
		time_elapsed_tot=0


	avg_speed=sum_speed/numdrones
	time_elapsed_tot+=numdrones/60

	font = pygame.font.SysFont("monospace", 24,bold=True)
	text = font.render("Time Elapsed (minutes): {0:.2f}".format(time_elapsed_tot), True, (100,100,100))
	screen.blit(text, (30, 670))
	text = font.render("Spill Area (K.M^2): {0:.2f}".format(actual_area), True, (100,100,100))
	screen.blit(text, (30, 640))
	text = font.render("Avg UAV Velocity (m/sec):): {0:.2f}".format(avg_speed), True, (100,100,100))
	screen.blit(text, (30, 610))
	text = font.render("(a). Flight Path of {} UAVs".format(numdrones), True, (000, 000, 000))
	screen.blit(text, (30, 700))
	pygame.draw.line(screen, (0,0,0), [50, 575],[150,575], 3)
	pygame.draw.line(screen, (0,0,0), [50, 565],[50,585], 3)
	pygame.draw.line(screen, (0,0,0), [150, 565],[150,585], 3)
	text = font.render("100px = 1K.m", True, (200, 000, 000))
	screen.blit(text, (30, 540))
	pygame.display.flip()

pygame.quit()
