import matplotlib.pyplot as plt 
import matplotlib 
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle

class HandlerLineWithFill(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        """
        Custom legend handler that supports:
          - Background fill (rectangle)
          - Linestyles ('-', '--', ':', etc.)
          - Single marker at the center
          - Alpha for the line itself
          
        orig_handle should be a tuple:
            (fill_color, line_color, linestyle, marker, line_alpha)
        """
        fill_color, line_color, linestyle, marker, line_alpha = orig_handle
        artists = []

        # Optional filled rectangle
        if fill_color is not None and str(fill_color).lower() != "none":
            patch = Rectangle(
                (x0, y0), width, height,
                facecolor=fill_color, alpha=0.3, transform=trans
            )
            artists.append(patch)

        # Optional line
        if line_color is not None and str(line_color).lower() != "none":
            line = Line2D(
                [x0, x0 + width], [y0 + height/2] * 2,
                color=line_color, linestyle=linestyle,
                linewidth=3.0, alpha=line_alpha,
                transform=trans
            )
            artists.append(line)

            # Optional single marker at center
            if marker is not None and str(marker).lower() != "none":
                marker_line = Line2D(
                    [x0 + width/2], [y0 + height/2],
                    color=line_color, marker=marker,
                    markersize=fontsize * 0.3,
                    markerfacecolor=line_color,
                    markeredgecolor=line_color,
                    linestyle="None", alpha=line_alpha,
                    transform=trans
                )
                artists.append(marker_line)

        return artists

def plot_airfoil(fig:matplotlib.figure.Figure, 
                 ax:matplotlib.axes.Axes,
                 plotting_elements:list, 
                 kwargs_list:list, 
                 legend_handle:list, 
                 legend_labels:list):
    # plotting elements shoudl be 
    # [xc, zcu, zcl, zcu_std (optional), zcl_std (optional)]
    # plotting_styles
    # [[], []]
    
    for element, kwargs_pair in zip(plotting_elements, kwargs_list):    
        # Plot mean airfoil  
        ax.plot(element[0]['xc'], element[0]['upper'], **kwargs_pair[0])
        ax.plot(element[0]['xc'], element[0]['lower'], **kwargs_pair[0])
        
        if element[1]['upper_bound'] is not None: 
            # Upper surface 2 sigma range 
            ax.fill_between(
                element[0]['xc'], 
                element[0]['upper'] + element[1]['upper_bound'],
                element[0]['upper'] - element[1]['upper_bound'],
                zorder=0,
                **kwargs_pair[1]
            )
            # Lower surface 2 sigma range 
            ax.fill_between(
                element[0]['xc'], 
                element[0]['lower'] - element[1]['lower_bound'],
                element[0]['lower'] + element[1]['lower_bound'],
                zorder=0,
                **kwargs_pair[1]
            )
    
    # Set up labels 
    ax.set_xlabel('Chordwise location, $x/c$')
    ax.set_ylabel('Thickness, $y/c$')
    
    # Set up legend 
    leg = ax.legend(legend_handle, legend_labels, handler_map={tuple: HandlerLineWithFill()}) 
    return fig, ax, leg


def plot_cp(fig:matplotlib.figure.Figure, 
            ax:matplotlib.axes.Axes,
            plotting_elements:list, 
            kwargs_list:list, 
            legend_handle:list, 
            legend_labels:list):
    """
    Plots Cp distribution 
    """
    
    # This is effectively same task as plotting the airfoil but with the axis inverted and the label changed
    f, ax, leg = plot_airfoil(fig, ax, plotting_elements, kwargs_list, legend_handle, legend_labels)
    ax.set_ylabel('Pressure coefficient, $C_p$')
    ax.invert_yaxis()
    return f, ax, leg

def plot_residual(fig:matplotlib.figure.Figure, 
            ax:matplotlib.axes.Axes,
            plotting_elements:list, 
            kwargs_list:list, 
            legend_handle:list, 
            legend_labels:list):
    """
    Plots Cp distribution 
    """
    
    # This is effectively same task as plotting the airfoil but with the axis inverted and the label changed
    for element, kwargs_pair in zip(plotting_elements, kwargs_list):    
        # Plot mean airfoil  
        ax.plot(element[0]['xc'], element[0]['upper'], **kwargs_pair[0])
        ax.plot(element[0]['xc'], element[0]['lower'], **kwargs_pair[0])
        
        if element[1]['upper_bound'] is not None: 
            # Upper surface 2 sigma range 
            ax.fill_between(
                element[0]['xc'], 
                element[0]['upper'] + element[1]['upper_bound'],
                element[0]['upper'] - element[1]['upper_bound'],
                **kwargs_pair[1]
            )
            
            # Lower surface 2 sigma range 
            ax.fill_between(
                element[0]['xc'], 
                element[0]['lower'] + element[1]['lower_bound'],
                element[0]['lower'] - element[1]['lower_bound'],
                **kwargs_pair[1]
            )
    
    # Set up labels 
    ax.set_xlabel('Chordwise location, $x/c$')
    ax.set_ylabel('Thickness, $y/c$')
    
    # Set up legend 
    leg = ax.legend(legend_handle, legend_labels, handler_map={tuple: HandlerLineWithFill()}) 
    return fig, ax, leg
