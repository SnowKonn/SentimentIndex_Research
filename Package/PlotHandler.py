import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
from matplotlib.dates import HourLocator, DateFormatter
import os
import matplotlib.font_manager as fm

def unique(list):
    x = np.array(list)
    return np.unique(x)



IBK_COLOR_LIST = ['#003592', '#000000', '#538fff', '#4c4c4c', '#9accff', '#808080', '#c9dcff', '#b3b3b3']
IBK_COLOR_BLUE = ['#003592',  '#538fff',  '#9accff',  '#c9dcff']
IBK_COLOR_BLACK = ['#000000', '#4c4c4c', '#808080', '#b3b3b3']

def plot_single(data, cols=None, spacing=.1, colors=None, loc=0, unit='', **kwargs):
    from pandas.plotting._matplotlib.style import get_standard_colors

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    if colors is None:
        colors = get_standard_colors(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    data_max_y_val = max(data.loc[:, cols[0]])
    data_min_y_val = min(data.loc[:, cols[0]])
    y_ticks = unit_info(data_min_y_val, data_max_y_val, n_ticks_approx=5, seq=True)
    plt.xlabel('')
    plt.gca().spines['top'].set_visible(False)  # 위 테두리 제거
    plt.gca().spines['right'].set_visible(False)  # 위 테두리 제거
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    ax.xaxis.set_major_formatter(DateFormatter('%m.%y'))
    plt.yticks(y_ticks)
    # ax.set_ylabel(ylabel=cols[0])
    ax.set_ylim([y_ticks[0], y_ticks[-1]])
    ax.set_xlim([data.index[0], data.index[-1]])
    # ax.tick_params(axis='x', labelrotation=0)

    lines, labels = ax.get_legend_handles_labels()
    plt.text(data.index[0] - (data.index[-1] - data.index[0]) * 0.1,
             y_ticks[-1] + (y_ticks[-1] - y_ticks[0]) * 0.1,
             unit)

    ax.legend(lines, labels, loc=loc, frameon=False)
    return ax


# Plot functions
def plot_multi(data: pd.DataFrame, cols=None, spacing=.1, colors=None, loc=0, unit_list=['', ''], date_format=False, **kwargs):

    from pandas.plotting._matplotlib.style import get_standard_colors
    
    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    if colors is None:
        colors = get_standard_colors(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    data_max_y_val = np.max(data.loc[:, cols[0]])
    data_min_y_val = np.min(data.loc[:, cols[0]])
    y_ticks = unit_info(data_min_y_val, data_max_y_val * 1.2, n_ticks_approx=6, seq=True)
    plt.xlabel('')
    plt.gca().spines['top'].set_visible(False)  # 위 테두리 제거
    plt.tick_params(axis='x', direction='in', which='both')
    plt.tick_params(axis='y', direction='in')
    if date_format:
            date_x_ticks = unit_info_date(data.index[0], data.index[-1], 6)
            plt.xticks(date_x_ticks)
            plt.gca().xaxis.set_major_formatter(DateFormatter('%y.%m'))
    
    plt.yticks(y_ticks)
    
    # ax.set_ylabel(ylabel=cols[0])
    plt.minorticks_off()
    ax.set_ylim([y_ticks[0], y_ticks[-1]])
    ax.set_xlim([data.index[0], data.index[-1]])
    ax.tick_params(axis='x', labelrotation=0)
    # ax.axis["bottom"].major_ticklabels.set_ha("center")
    plt.text(data.index[0] -(data.index[-1] - data.index[0]) * 0.1,
             y_ticks[-1] + (y_ticks[-1] - y_ticks[0]) * 0.07,
             unit_list[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data_max_y_val = np.max(data.loc[:, cols[n]])
        data_min_y_val = np.min(data.loc[:, cols[n]])
        y_ticks = unit_info(data_min_y_val, data_max_y_val * 1.2, n_ticks_approx=6, seq=True)
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        plt.minorticks_off()
        plt.xlabel('')
        
        plt.yticks(y_ticks)
        plt.gca().spines['top'].set_visible(False)  # 위 테두리 제거
        plt.tick_params(axis='x', direction='in', which='both')
        plt.tick_params(axis='y', direction='in')
        ax.tick_params(axis='x', labelrotation=0)
                
        if date_format:
                date_x_ticks = unit_info_date(data.index[0], data.index[-1], 6)
                plt.xticks(date_x_ticks)
                plt.gca().xaxis.set_major_formatter(DateFormatter('%y.%m'))
        
        ax_new.set_xlim([data.index[0], data.index[-1]])
        ax_new.set_ylim([y_ticks[0], y_ticks[-1]])
        plt.text(data.index[-1] + (data.index[-1] - data.index[0]) * 0.01,
                 y_ticks[-1] + (y_ticks[-1] - y_ticks[0]) * 0.07,
                 unit_list[1])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=loc, frameon=False, ncol=min(4, len(labels)))
    return ax



def draw_date_coloured_scatterplot(etfs, prices):
    """
    Create a scatterplot of the two ETF prices, which is
    coloured by the date of the price to indicate the
    changing relationship between the sets of prices
    """
    # Create a yellow-to-red colourmap where yellow indicates
    # early dates and red indicates later dates
    plen = len(prices)
    colour_map = plt.cm.get_cmap('YlGnBu')
    colours = np.linspace(0.1, 1, plen)

    # Create the scatterplot object
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]],
        s=5, c=colours, cmap=colour_map,
        edgecolor='k', alpha=0.8, linewidth=0.5
    )

    # Add a colour bar for the date colouring and set the
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen // 9].index]
    )
    plt.xlabel(prices.columns[0])
    plt.ylabel(prices.columns[1])
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.gca().spines['right'].set_visible(False)  # 오른쪽 테두리 제거
    plt.gca().spines['top'].set_visible(False)  # 위 테두리 제거
    plt.show()


def cal_between_logic(a, lower_bound, upper_bound, equal=True, equal_type='all'):
    if equal:
        if equal_type == 'all':
            if (a >= lower_bound) and (a <= upper_bound):
                return True
            else:
                return False
        elif equal_type == 'left':
            if (a >= lower_bound) and (a < upper_bound):
                return True
            else:
                return False
        elif equal_type == 'right':
            if (a > lower_bound) and (a <= upper_bound):
                return True
            else:
                return False
    else:
        if (a > lower_bound) and (a < upper_bound):
            return True
        else:
            return False


def unit_info(y_min, y_max, n_ticks_approx, unit_list=None, seq=False):
    if unit_list is None:
        unit_list = [1, 2, 2.5, 5, 10]
    interval = y_max - y_min

    unit_raw = interval / (n_ticks_approx - 2)
    for i in np.linspace(-5, 20, 26):
        quotient_digits = unit_raw // 10 ** i
        unit_condition = cal_between_logic(quotient_digits, unit_list[0], unit_list[-1], equal_type='left')
        if unit_condition:
            n_digits = i
            unit_distance_list = list(map(lambda x: abs(x - quotient_digits), unit_list))
            min_distance_unit = min(unit_distance_list)
            unit_index = unit_distance_list.index(min_distance_unit)
            unit = unit_list[unit_index]

    if seq:
        unit_final = unit * 10 ** n_digits
        result = np.arange(start=y_min // unit_final * unit_final,
                           stop=((y_max // unit_final) + 2) * unit_final,
                           step=unit_final)
        return result
    else:
        return unit, n_digits


def unit_info_date(y_min, y_max, n_ticks_approx, unit_list=None):
    if unit_list is None:
        unit_list = [1, 2, 3, 6, 12, 24]
    y_min = y_min - np.timedelta64(y_min.day -1,'D')
    interval_month = (y_max - y_min)/(np.timedelta64(1,'W')*4)

    unit_raw = interval_month / (n_ticks_approx)
    interval_month_revised = np.array(unit_list)[
        abs(np.array(unit_list) - unit_raw) == min(abs(np.array(unit_list) - unit_raw))][0]
    sub_interval =  unit_list[max(unit_list.index(interval_month_revised) - 1, 0)]
    date_time_index = pd.date_range(start=y_min + (np.timedelta64(sub_interval,'W')*4), end=y_max, freq=str(interval_month_revised)+'MS')
    
    return date_time_index
    