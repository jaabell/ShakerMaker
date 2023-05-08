from shakermaker import shakermaker
from shakermaker.station import Station
from shakermaker.tools.plotting import ZENTPlot
import matplotlib.pyplot as plt
import glob

files = glob.glob("dk*.npz")


fig = plt.figure(1)

for f in files:
	s = Station()

	s.load(f)

	# Visualize results

	ZENTPlot(s, show=False, xlim=[0,15], fig=fig, label=f)

plt.legend()


plt.show()