import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import OrderedDict
from matplotlib.animation import FuncAnimation 
from gym import utils
import sys
from six import StringIO, b
from IPython.display import HTML

COLORS = OrderedDict([
    (b"W" , [160, 160, 160]),
    (b"S" , [224, 224, 224]),
    (b"E" , [224, 224, 224]),
    (b"F" , [255, 0, 0]),
    (b"H" , [20, 20, 20]),
    (b"G" , [50, 255, 50]),
    (b"P" , [51, 150, 255])
])

class Renderer:
    """
        Rendering class for the gridworld environment. Three rendering
        options are available; plot rendering, render and store(buffer)
        and stdout. Plot rendering renders the grid whenever the render
        method is called while buffer rendering stores the grid's rendered 
        image in the buffer for future animations. String(stdout) rendering
         simply prints the grid or return the string at each call.

        Args:
            gridmap: Numpy array of char datatype of the grid map.
        
        Methods:
            visual_render: Directly renders the grid to the screen.
            buffer_render: Renders the grid map and stores it.
            animate: Animates the images in the buffer. Three modes
                are available namely plot, js and html. Plot mode animates
                and renders on the screen while js and html return js or
                html files mostly for ipython.
            analysis: Animates the value distribution and the greedy
                policy constructed from the values in the buffer.
    """

    def __init__(self, gridmap):
        self.gridmap = gridmap
        ncolors = len(COLORS)
        tile_to_int = {c: i for i, c in zip(range(ncolors), COLORS.keys())}
        self.background = np.array([tile_to_int[tile] for tile in self.gridmap.ravel()])
        self.background = self.background.reshape(self.gridmap.shape)

        bounds = np.linspace(0, len(COLORS), num=len(COLORS)+1)
        self.norm = BoundaryNorm(bounds, len(COLORS))
        self.cmap = ListedColormap([[v/255 for v in rgb] for rgb in COLORS.values()])

        self.reset_buffer()

    def visaul_render(self, state):
        img = np.copy(self.background)
        img[state] = 7
        try:
            self.plot.set_array(img)
        except AttributeError:
            self.reset_figure()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    
    def buffer_render(self, state, info=None):
        img = np.copy(self.background)
        img[state] = 7
        self.frames.append(img if info is None else (img, info))

    def reset_figure(self):
        plt.ion()
        plt.axis('off')
        self.figure, self.ax = plt.subplots()
        self.plot = self.ax.imshow(self.background, cmap=self.cmap, norm=self.norm)

    def reset_buffer(self):
        self.frames = []

    def animate(self, mode="js"):
        plt.ioff()
        heigth, width = self.background.shape
        ratio = width/heigth
        figure, ax = plt.subplots(figsize=(3*ratio,3))
        im = plt.imshow(self.background, cmap=self.cmap, norm=self.norm, animated=True)
        title = ax.text(0.5,0.90, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
        ax.axis('off')

        def update(i):
            data = self.frames[i]
            if isinstance(data, tuple):
                img, text = data
                title.set_text(text)
            else:
                img = data
            im.set_array(img)
            return im, title

        ani = FuncAnimation(figure, update, frames=len(self.frames), interval=1000/60, blit=True, repeat=False)
        if mode == "html":
            return HTML(ani.to_html5_video())
        elif mode == "js":
            return HTML(ani.to_jshtml())
        elif mode == "plot":
            plt.show()

    def string_render(self, state, mode):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = state
        desc = self.gridmap.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")
        if mode == 'ansi':
                return outfile

    def analysis(self, value_buffer, mode="notebook"):
        plt.ioff()
        heigth, width = self.background.shape
        ratio = width/heigth
        figure, ax = plt.subplots(figsize=(3*ratio,3))
        X, Y = np.meshgrid(np.arange(heigth), np.arange(width))
        ax.axis('off')


        plt.imshow(self.background, cmap=self.cmap, norm=self.norm, animated=False)
        im = plt.imshow(np.zeros(shape=self.background.shape), cmap="Blues", vmin=0, vmax=1, animated=True, alpha=0.5)

        coords = np.argwhere(self.background != 0)
        quiver = plt.quiver(coords[:, 1], coords[:, 0], *([np.ones(shape=X.shape)]*2))

        arr_u = np.array([-1, 0, 1, 0])
        arr_v = np.array([0, 1, 0, -1])

        def update(i):
            values = np.array([max(value_buffer[i][(x, y)]) for (x, y) in zip(X.ravel(), Y.ravel())])
            actions = np.array([max(range(4), key=lambda a: value_buffer[i][(x, y)][a]) for (x, y) in coords])

            quiver_u = arr_u[actions]
            quiver_v = arr_v[actions]

            quiver.set_UVC(quiver_u, quiver_v)

            im.set_array(values.reshape(X.shape).transpose())
            return im,

        ani = FuncAnimation(figure, update, frames=len(value_buffer), interval=1000/60, blit=True, repeat=True)

        if mode == "notebook":
            return HTML(ani.to_jshtml())
        elif mode == "plot":
            plt.show()

def animater(buffer, mode="js"):
    """ Animates the buffer for three modes.
    """
    plt.ioff()
    heigth, width, _ = buffer[0].shape
    ratio = width/heigth
    figure, ax = plt.subplots(figsize=(4*ratio,4))
    im = plt.imshow(buffer[0])
    ax.axis('off')

    def update(i):
        im.set_array(buffer[i])
        return im,

    ani = FuncAnimation(figure, update, frames=len(buffer), interval=1000/60, blit=True, repeat=False)
    if mode == "html":
        return HTML(ani.to_html5_video())
    elif mode == "js":
        return HTML(ani.to_jshtml())
    elif mode == "plot":
        plt.show()

def comparison(*log_name_pairs, texts=[[""]*3]):
    """ Plots the given logs. There will be as many plots as
    the length of the texts argument. Logs will be plotted on
    top of each other so that they can be compared. For each
    log, mean value is plotted and the area between the
    +std and -std of the mean will be shaded.
    """
    plt.ioff()
    plt.close()

    def plot_texts(title, xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    for i, (title, xlabel, ylabel) in enumerate(texts):
        for logs, name in log_name_pairs:
            smoothed_logs = np.stack([smoother([x[i] for x in log], 7) for log in logs])
            std_logs = np.std(smoothed_logs, axis=0)
            mean_logs = np.mean(smoothed_logs, axis=0)
            max_logs = np.max(smoothed_logs, axis=0)
            min_logs = np.min(smoothed_logs, axis=0)
            plot_texts(title, xlabel, ylabel)
            plt.plot(mean_logs, label=name)
            plt.legend()
            plt.fill_between(np.arange(len(mean_logs)), np.minimum(mean_logs+std_logs, max_logs), np.minimum(mean_logs-std_logs, min_logs), alpha=0.4)
        
        plt.show()

def smoother(array, ws):
    return np.array([sum(array[i:i+ws])/ws for i in range(len(array) - ws)])