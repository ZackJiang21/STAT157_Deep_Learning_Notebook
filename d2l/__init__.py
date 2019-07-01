from IPython import display
from matplotlib import pyplot as plt


def use_svg_display():
    # Use the svg format to display plot in jupyter
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    # Change the default figure size
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
