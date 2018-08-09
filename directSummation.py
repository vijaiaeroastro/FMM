"""
A Program to compute and plot N Body simulation for learning purposes (Following Prof. Barba's Group)
@author : Vijai Kumar
@email  : vijai@vijaikumar.in
"""

# Import the necessary libraries
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from funcy import flatten
import datetime
import time
import multiprocessing

# Matplotlib Globals
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# A simple data structure for storing the point
class Point():
    """
    A Simple point structure that holds a point or could create a random point
    """
    def __init__(self, coordinates=list(), domain=1.0):
        self.domain = domain
        if coordinates:
            self.x = coordinates[0]
            self.y = coordinates[1]
            self.z = coordinates[2]
        else:
            self.x = self.domain * np.random.random()
            self.y = self.domain * np.random.random()
            self.z = self.domain * np.random.random()

    def getNumpyArray(self):
        return np.array([self.x,self.y,self.z])

    def distance(self, other):
        xDiff = other.x - self.x
        xSquared = xDiff * xDiff
        yDiff = other.y - self.y
        ySquared = yDiff * yDiff
        zDiff = other.z - self.z
        zSquared = zDiff * zDiff
        dist = np.sqrt(xSquared + ySquared + zSquared)
        return dist

    def __str__(self):
        return ("{0}(({1},{2},{3}),{4})".format(self.__class__.__name__, self.x, self.y, self.z, self.domain))

# A Particle class
class Particle(Point):
    """
    A Simple structure to hold a particle and all of its neighbours
    """
    def __init__(self, coordinates=list(), domain=1.0, mass=0.1, idx=None):
        Point.__init__(self, coordinates=coordinates, domain=domain)
        self.m = mass
        self.domain = domain
        self.Neighbours = list()
        self.idx = idx
        self.phi = 0.0

    def getId(self):
        return self.idx

    def addNeighbour(self, Neighbour):
        self.Neighbours.append(Neighbour)

    def getNeighBours(self):
        return self.Neighbours
    
    def getNeighBoursCount(self):
        return len(self.Neighbours)

    def __str__(self):
        return ("{0}({1},({2}/))")

# Create random particles
def createParticles(n):
    particles = list()
    for i in range(1,n):
        particle = Particle(mass=(1.0/i),idx=i)
        particles.append(particle)
    return particles

# Compute neighbour for 1 particle
def findNeighbourForSingleParticle(currentParticle, particleArray,tolerance):
    for part in particleArray:
        if part != currentParticle:
            distance = currentParticle.distance(part)
            if (distance < tolerance) or (distance == tolerance):
                currentParticle.addNeighbour(part)

# Multithreaded version of compute neighbours (Not really required in this code)
def createParticlesAndComputeNeighbours(n,tolerance):
    particles = createParticles(n)
    jobs = list()
    for particle in particles:
        currentParticle = particle
        p = multiprocessing.Process(target=findNeighbourForSingleParticle, args=(currentParticle,particles,tolerance))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    return particles

# Compute potential for 1 particle
def directSummationForSingleParticle(currentTarget,particleList,sharedVariable):
    for source in particleList:
        if source != currentTarget: # To avoid self contribution
            radius = currentTarget.distance(source)
            currentTarget.phi = currentTarget.phi + (source.m/radius)
    sharedVariable.append(currentTarget.phi)

# Direct summation code (Multithreaded)
def directSummation(particleList):
    jobs = list()
    manager = multiprocessing.Manager()
    sharedVariable = manager.list()
    for target in particleList:
        currentTarget = target
        p = multiprocessing.Process(target=directSummationForSingleParticle, args=(currentTarget,particleList,sharedVariable))
        jobs.append(p)
        p.start()
        for proc in jobs:
            proc.join()
    targetCounter = 0
    for target in particleList:
        target.phi = sharedVariable[targetCounter]
        targetCounter = targetCounter + 1

# Plot the N body result
def plotParticles(particles):
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.subplots_adjust(top=0.986,bottom=0.014,left=0.007,right=0.993,hspace=0.2,wspace=0.2)
    # left plot
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.scatter([particle.x for particle in particles], 
            [particle.y for particle in particles], 
            [particle.z for particle in particles], s=30, c='b')
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)
    ax.set_xlabel(r'$X Axis$')
    ax.set_ylabel(r'$Y Axis$')
    ax.set_zlabel(r'$Z Axis$')
    ax.set_title(r'$\textsc{Particle Distribution}$')
    # right plot
    ax = fig.add_subplot(1,2,2, projection='3d')
    scale = 50   # scale for dot size in scatter plot
    ax.scatter([particle.x for particle in particles], 
            [particle.y for particle in particles], 
            [particle.z for particle in particles],
            s=np.array([particle.phi for particle in particles])*scale, c='b')
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)
    ax.set_xlabel(r'$X Axis$')
    ax.set_ylabel(r'$Y Axis$')
    ax.set_zlabel(r'$Z Axis$')
    ax.set_title(r'\textsc{Particle Distribution (Radius implies potential)}')
    plt.savefig('potential_direct_summation.pdf')

if __name__ == "__main__":
    numberOfParticles = 100
    tolerance = 0.15
    particles = createParticlesAndComputeNeighbours(numberOfParticles,0.15)
    directSummation(particles)
    plotParticles(particles)