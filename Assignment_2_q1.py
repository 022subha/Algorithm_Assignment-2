#!/usr/bin/env python
# coding: utf-8

# # Polygon Triangulation Problem

# In[57]:


from shapely.geometry import Point, Polygon
from matplotlib import pyplot as plt
from typing import List
from math import sqrt
import sys

import random

class polygon:
    def __init__(self, n):
        self.poly  = Polygon()
        self.sides = n
        self.range = 50

        if(n*10 > self.range):
            self.range = n*10
    
    def generate(self):
        random.SystemRandom()
        x = random.sample(range(-self.range, self.range), self.sides)
        y = random.sample(range(-self.range, self.range), self.sides)
        z = list(zip(x,y))
        self.poly = Polygon(z)
        self.poly = self.poly.convex_hull
        n = len(self.poly.exterior.xy[0])-1
        while n <self.sides:
            random.SystemRandom()
            x, y   = (random.randint(-self.range, self.range),
                      random.randint(-self.range, self.range))

            x1, y1 = self.poly.exterior.xy

            x1.append(x)
            y1.append(y)
            z = list(zip(x1, y1))
            self.poly = Polygon(z)
            self.poly = self.poly.convex_hull
            n = len(self.poly.exterior.xy[0])-1
      
    def plot(self):
        x, y = self.poly.exterior.xy
        plt.plot(x, y)
        plt.show()

    def getsides(self):
        return self.sides


# In[58]:


def dist(p1, p2):
    return sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]))

def cost(vertices, x, y, z):

    distance = dist(vertices[x],vertices[y])+dist(vertices[y],vertices[z])+dist(vertices[z],vertices[x])
    return distance


# # Brute Force Approach

# In[59]:


def brute_force_MWT(vertices, i, j):  
    
    if(j < i+2):
        return 0

    res = 1000000.0
    for k in range (i+1, j):
        res = min(res,(brute_force_MWT(vertices, i, k) + brute_force_MWT(vertices, k, j) + cost(vertices, i, k, j)))


    return round(res,4)


# # Dynamic Programming Approach 

# In[60]:


def dynam_progr_MWT(vertices):
    n = len(vertices)

    T = [[0.0]*n for _ in range(n)]
    for diagonal in range(n):
        i = 0
        for j in range(diagonal, n):
            if j >= i + 2:
                T[i][j] = sys.maxsize
                for k in range(i+1, j):
                    weight  = dist(vertices[i], vertices[j]) +                              dist(vertices[j], vertices[k]) +                              dist(vertices[k], vertices[i])

                    T[i][j] = min(T[i][j], weight+T[i][k]+T[k][j])
            i+=1

    return T[0][-1]


# # Greedy Programming Approach

# In[61]:


import math
def position(arr):
    convex = max(arr, key = lambda x: (x[1], x[0]))
    for i in range(len(arr)):
        if arr[i] == convex:
            return i

def inwards(x,y,z):
    angle = math.degrees(math.atan2((z[1]-y[1]),(z[0]-y[0]))-math.atan2((z[1]-y[1]),(z[0]-y[0])))
    if angle >= 180:
        return False
    else:
        return True
        
def inList(x,y,z):
    if   x in y:
        return 1
    elif x in z:
        return 2

def perimeter(x,y,z):
    return (dist(x,y)+dist(y,z)+dist(z,x))


# In[62]:


def greed_progr_MWT(vertices):

    n   = len(vertices)
    div = position(vertices)+1

    L   = vertices[:div]
    R   = vertices[div:]
    print(L)
    print(R)
    vertices_merged = L+R
    vertices_merged = sorted(vertices_merged, key = lambda k: (k[1],k[0]), reverse=True)
    print(vertices_merged)

    L   = set(L)
    R   = set(R)
    results = []

    q = []
    q.append(vertices_merged[0])
    q.append(vertices_merged[1])

    last = 1
    for i in range(2,n-1):
        if inList(vertices_merged[i],L,R) == inList(vertices_merged[last],L,R):
            q.append(vertices_merged[i])
            last = i
            
            if(inwards(q[0],q[1],q[2])==True and len(q)>2):
                p1 = Point(q[0])
                p2 = Point(q[1])
                p3 = Point(q[2])
                temp_cost = p1.distance(p2)+p2.distance(p3)+p3.distance(p1)
                results.append(temp_cost)
                q.remove(q[1])
        else:
            temp = q[0]
            q.remove(q[0])

            while(len(q) >= 2):
                p1 = Point(q[0])
                p2 = Point(q[1])
                p3 = Point(vertices_merged[i])
                temp_cost = p1.distance(p2)+p2.distance(p3)+p3.distance(p1)
                results.append(temp_cost)
                q.remove(q[1])
            p1 = Point(temp)
            p2 = Point(q[0])
            p3 = Point(vertices_merged[i])
            temp_cost = p1.distance(p2)+p2.distance(p3)+p3.distance(p1)
            results.append(temp_cost)
            print(results)
            q.append(vertices_merged[i])
            last = i

    temp_cost = perimeter(vertices_merged[n-1],vertices_merged[n-2],vertices_merged[n-3])
    results.append(temp_cost)
    
    results = sorted(results, key=lambda x: x)
    print(results)
    return results[0]


# # Testing

# In[64]:


pol = polygon(5)
pol.poly = Polygon([(0, 0), (2, 0), (2, 1), (1,2), (0, 1)])
vx = [(0, 0), (2, 0), (2, 1), (1, 2), (0, 1)]
print(brute_force_MWT(vx,0,4))
print(dynam_progr_MWT(vx))
print(greed_progr_MWT(vx))
pol.plot()


# In[ ]:




