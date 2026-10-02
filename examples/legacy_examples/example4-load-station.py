from shakermaker import shakermaker
from shakermaker.station import Station
from shakermaker.tools.plotting import ZENTPlot

s = Station()

s.load("mystation.npz")

print(s)

# Visualize results
ZENTPlot(s, show=True)#, xlim=[0,3])
