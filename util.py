import numpy as np

EPS = 1e-10
def workstation_travel_time(ws1, ws2):
    return abs(ws1.x - ws2.x) + abs(ws1.y - ws2.y)
