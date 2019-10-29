import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import pickle

# Import data from pickle
filename = 'di_data2'
with open(filename, 'rb') as f:
        file_dict = pickle.load(f)
tol_0K = file_dict['tol_0K']
tol_at_T = file_dict['tol_at_T' ]
union_grid = file_dict['union_grid']
Ts = file_dict['Ts']
total_time = file_dict['time']
mp_plot_grid = file_dict['mp_plot_grid']
slbw_plot_grid = file_dict['slbw_plot_grid']


nice_font = True
choose_levels = True
levels = [1e-4, 1e-3, 1e-2, 1e-1]
styles = ['solid', 'dashed', 'dashdot', 'dotted']

scaling = 1.8
ratio = [10, 4]
figsize = [ratio[0]*scaling, ratio[1]*scaling]
mpl.rcParams['figure.figsize'] = figsize
if nice_font:
    rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rcParams.update({'font.size': 16})
    rc('text', usetex=True)

# Find values for colorbar range
max_color = np.min([np.max([mp_plot_grid.max(), slbw_plot_grid.max()]),int(1e5)])
min_color = np.max([np.min([mp_plot_grid.min(), slbw_plot_grid.min()]),1e-20])
print(min_color)

# Initialize figure
fig = plt.figure()

# Plot Multipole Error
plot_grid = mp_plot_grid
ax1 = fig.add_subplot(121)
grid1 = ax1.pcolormesh(union_grid, Ts, plot_grid, norm=colors.LogNorm(vmin=min_color, vmax=max_color), cmap='plasma')
if choose_levels:
    CS = ax1.contour(union_grid, Ts, plot_grid, levels, linestyles=styles, colors='k')
    # ax1.clabel(CS, levels, inline=1, fmt='%1.1e', fontsize=10)
else: 
    CS = ax1.contour(union_grid, Ts, plot_grid, colors='k')
    ax1.clabel(CS, fmt='%1.1e', fontsize=10)
ax1.set_xscale('log')
ax1.set_xlabel('E (eV)')
ax1.set_ylabel('T (Kelvin)')
ax1.set_title('Multipole Relative Error')
ax1.set_yscale('log')
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
# ax1.grid()

# Plot SLBW Error
plot_grid = slbw_plot_grid
ax2 = fig.add_subplot(122)
grid2 = ax2.pcolormesh(union_grid, Ts, plot_grid, norm=colors.LogNorm(vmin=min_color, vmax=max_color), cmap='plasma')
if choose_levels:
    CS = ax2.contour(union_grid, Ts, plot_grid, levels, linestyles=styles, colors='k')
    # ax2.clabel(CS, levels, inline=1, fmt='%1.1e', fontsize=10)
else: 
    CS = ax2.contour(union_grid, Ts, plot_grid, colors='k')
    ax2.clabel(CS, fmt='%1.1e', fontsize=10)
ax2.set_xscale('log')
ax2.set_xlabel('E (eV)')
ax2.set_ylabel('T (Kelvin)')
ax2.set_title('SLBW Relative Error')
ax2.set_yscale('log')
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
# ax2.grid()

fig.tight_layout(w_pad=1.0)

# Add color bar with axes [left, bottom, width, height]
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.90, 0.30, 0.05, 0.65])
norm=colors.LogNorm(vmin=min_color, vmax=max_color)
print('Attempted min/max for LogNorm color bar: {:E}/{:E}'.format(min_color, max_color))
print('---------------- \n \n')
# Make my own ticks
# Find powers of 10
min_pow = np.ceil(np.log10(min_color))
max_pow = np.floor(np.log10(max_color))
tick_locations = np.logspace(min_pow, max_pow, (max_pow - min_pow)+1)
colorbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap='plasma', norm=norm, ticks=tick_locations)
# colorbar = fig.colorbar(grid2, cax=cbar_ax)

# Colorbar testing junk below
# cbar_ax.xaxis.set_major_formatter(FormatStrFormatter(fmts[i]))
# test = cbar_ax.yaxis.get_ticklabels()
# test2 = cbar_ax.yaxis.get_ticklocs()
# test_scalarmappable = mpl.cm.ScalarMappable(norm=colors.LogNorm(vmin=min_color, vmax=max_color), cmap='plasma')
# test_scalarmappable = mpl.cm.ScalarMappable(norm=colors.Normalize(vmin=min_color, vmax=max_color), cmap='plasma')
# print('Tick Labels Should Show up here')
# print(test)
# print(test2)
# cbarlabels = np.logspace(np.log10(min_color), np.log10(max_color), num=5, endpoint=True)
# cbar_ax.yaxis.set_ticks(test2)
# cbar_ax.yaxis.set_ticklabels(test)
# cbar_ax.yaxis.set_major_formatter(FormatStrFormatter('%.0E'))

# Add contour legend [left, bottom, width, height]
contour_ax = fig.add_axes([0.90, 0.10, 0.05, 0.1])
legend_elements = [Line2D([0],[0],linestyle=styles[0], label=levels[0], color='k'),
                   Line2D([0],[0],linestyle=styles[1], label=levels[1], color='k'),
                   Line2D([0],[0],linestyle=styles[2], label=levels[2], color='k'),
                   Line2D([0],[0],linestyle=styles[3], label=levels[3], color='k')]
contour_ax.legend(handles=legend_elements, loc='center', title='Tolerance\n Contours')
contour_ax.set_axis_off()


title = 'Direct Integration - Energy Range: {:e} - {:e} eV \n 0K_tol: {:f} \n N_Points Union: {:d} \n N Temperatures: {:d} \n Time: {:f} min'.format(union_grid[0], union_grid[-1], tol_0K, len(union_grid), len(Ts), total_time)
# fig.suptitle(title)

fig.savefig('./figs/'+title+".pdf")
plt.show()

