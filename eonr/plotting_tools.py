# -*- coding: utf-8 -*-

import matplotlib.colors as colors
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import seaborn as sns

plt.style.use('ggplot')


class Plotting_tools(object):
    def __init__(self, EONR):
        self.df_data = EONR.df_data
        self.df_delta_tstat = EONR.df_delta_tstat
        self.cost_n_fert = EONR.cost_n_fert
        self.cost_n_social = EONR.cost_n_social
        self.costs_fixed = EONR.costs_fixed
        self.price_grain = EONR.price_grain
        self.price_ratio = EONR.price_ratio
        self.col_n_app = EONR.col_n_app
        self.col_yld = EONR.col_yld
        self.col_crop_nup = EONR.col_crop_nup
        self.col_n_avail = EONR.col_n_avail
        self.unit_currency = EONR.unit_currency
        self.unit_fert = EONR.unit_fert
        self.unit_grain = EONR.unit_grain
        self.unit_area = EONR.unit_area
        self.unit_rtn = EONR.unit_rtn
        self.unit_nrate = EONR.unit_nrate
        self.model = EONR.model
        self.model_temp = EONR.model_temp
        self.ci_level = EONR.ci_level
        self.base_dir = EONR.base_dir
        self.base_zero = EONR.base_zero
        self.print_out = EONR.print_out
        self.location = EONR.location
        self.year = EONR.year
        self.time_n = EONR.time_n
        self.onr_name = EONR.onr_name
        self.onr_acr = EONR.onr_acr

        self.R = EONR.R
        self.coefs_grtn = EONR.coefs_grtn
        self.coefs_grtn_primary = EONR.coefs_grtn_primary
        self.coefs_nrtn = EONR.coefs_nrtn
        self.coefs_social = EONR.coefs_social
        self.results_temp = EONR.results_temp

        self.mrtn = EONR.mrtn
        self.eonr = EONR.eonr
        self.df_ci = EONR.df_ci
        self.df_ci_pdf = EONR.df_ci_pdf
        self.df_delta_tstat = EONR.df_delta_tstat
        self.df_der = EONR.df_der
        self.df_linspace = EONR.df_linspace

        self.fig_delta_tstat = EONR.fig_eonr
        self.fig_derivative = EONR.fig_derivative
        self.fig_eonr = EONR.fig_eonr
        self.fig_tau = EONR.fig_tau
        self.ci_list = EONR.ci_list
        self.alpha_list = EONR.alpha_list
        self.df_results = EONR.df_results
        self.bootstrap_ci = EONR.bootstrap_ci
        self.metric = EONR.metric
        self.palette = self._seaborn_palette(color='muted', cmap_len=10)

    #  Following are plotting functions
    def _add_labels(self, g, x_max=None, run_n=None, show_model=False,
                    fert_range=False):
        '''
        Adds EONR and economic labels to the plot
        '''
        if run_n is None:
            df_ci = self.df_ci[self.df_ci['run_n'] ==
                               self.df_ci['run_n'].max()]
        else:
            df_ci = self.df_ci[self.df_ci['run_n'] == run_n]

        if x_max is None:
            _, x_max = g.fig.axes[0].get_xlim()
        boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
        el = mpatches.Ellipse((0, 0), 0.3, 0.3, angle=50, alpha=0.5)
        g.ax.add_artist(el)
        label_eonr = '{0}: {1:.0f} {2}\nMRTN: ${3:.2f}'.format(
                self.onr_acr, self.eonr, self.unit_nrate, self.mrtn)
        if self.eonr <= x_max:
            if fert_range is True:
                err_l = [self.eonr -
                         df_ci[df_ci['level'] == 0.5]['pl_l'].item()]
                err_u = [df_ci[df_ci['level'] == 0.5]['pl_u'].item() -
                         self.eonr]
                g.ax.errorbar(x=self.eonr, y=self.mrtn,
                              xerr=[err_l, err_u], marker='.', markersize=15,
                              color=self.palette[2], markeredgecolor='white',
                              ecolor='#7b7b7b', capsize=6, capthick=1.5,
                              label=label_eonr)
            else:
                g.ax.plot([self.eonr], [self.mrtn], marker='.', markersize=15,
                          color=self.palette[2], markeredgecolor='white',
                          label=label_eonr)

        if self.eonr > 0 and self.eonr < self.df_data[self.col_n_app].max():
            g.ax.annotate(
                label_eonr,
                xy=(self.eonr, self.mrtn), xytext=(-80, -30),
                textcoords='offset points', ha='left', va='top', fontsize=8,
                color='#444444',
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5)),
                arrowprops=dict(arrowstyle='-|>',
                                color="grey",
                                patchB=el,
                                shrinkB=10,
                                connectionstyle='arc3,rad=-0.3'))
        else:
            g.ax.annotate(
                label_eonr,
                xy=(0, 0), xycoords='axes fraction', xytext=(0.98, 0.05),
                ha='right', va='bottom', fontsize=8, color='#444444',
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5)))

        label_econ = ('Grain price: ${0:.2f}\nN fertilizer cost: ${1:.2f}'
                      ''.format(self.price_grain, self.cost_n_fert))

        if self.costs_fixed != 0.0:
            label_econ = ('{0}\nFixed costs: ${1:.2f}'
                          ''.format(label_econ, self.costs_fixed))
        if self.cost_n_social > 0 and self.metric is True:
            label_econ = ('{0}\nSocial cost of N: ${1:.2f}\nPrice ratio: '
                          '{2:.2f}'.format(label_econ, self.cost_n_social,
                                           self.price_ratio))
        elif self.cost_n_social > 0 and self.metric is False:
            label_econ = ('{0}\nSocial cost of N: ${1:.2f}\nPrice ratio: '
                          '{2:.3f}'.format(label_econ, self.cost_n_social,
                                           self.price_ratio))
        elif self.cost_n_social == 0 and self.metric is True:
            label_econ = ('{0}\nPrice ratio: {1:.2f}'
                          ''.format(label_econ, self.price_ratio))
        else:
            label_econ = ('{0}\nPrice ratio: {1:.3f}'
                          ''.format(label_econ, self.price_ratio))

        if self.base_zero is True:
            label_econ = ('{0}\nBase zero: {1}{2:.2f}'.format(
                    label_econ, self.unit_currency,
                    self.coefs_grtn_primary['b0'].n))
        if show_model is True:
            label_econ = ('{0}\nModel: {1}'.format(
                    label_econ, self.model_temp))
        g.ax.annotate(
            label_econ,
            xy=(0, 1), xycoords='axes fraction', xytext=(0.98, 0.95),
            horizontalalignment='right', verticalalignment='top',
            fontsize=7, color='#444444',
            bbox=dict(boxstyle=boxstyle_str,
                      fc=(1, 1, 1), ec=(0.5, 0.5, 0.5), alpha=0.75))
        return g

    def _add_title(self, g, size_font=12, der=False):
        '''
        Adds the title to the plot
        '''
        title_text = ('{0} {1} - {2} N Fertilizer Timing'
                      ''.format(self.year, self.location, self.time_n))
        if der is True:
            divider = make_axes_locatable(g.axes[0])
        else:
            divider = make_axes_locatable(g.ax)
        cax = divider.append_axes("top", size="15%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor('#7b7b7b')
        text_obj = AnchoredText(title_text, loc=10, frameon=False,
                                prop=dict(backgroundcolor='#7b7b7b',
                                          size=size_font, color='white',
                                          fontweight='bold'))
        if der is True:
            size_font = self._set_title_size(g, title_text, size_font,
                                             coef=1.15)
        else:
            size_font = self._set_title_size(g.fig, title_text, size_font,
                                             coef=1.15)
        text_obj.prop.set_size(size_font)
        cax.add_artist(text_obj)
        return g

    def _add_ci(self, g, ci_type='profile-likelihood', ci_level=None):
        '''
        Adds upper and lower confidence curves to the EONR plot
        <ci_type> --> type of confidence interval to display
            'profile-likelihood' (default) is the profile likelihood confidence
            interval (see Cook and Weisberg, 1990)
            'wald' is the Wald (uniform) confidence interval
            if the profile-likelihood confidence interval is available, it is
            recommended to use this for EONR data
        <ci_level> --> the confidence level to display
        '''
        msg = ('Please choose either "profile-likelihood" or "wald" for '
               '<ci_type>')
        ci_types = ['profile-likelihood', 'wald', 'bootstrap']
        assert ci_type.lower() in ci_types, msg
        if ci_level is None:
            ci_level = self.ci_level
        last_run_n = self.df_ci['run_n'].max()
        df_ci_last = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
                                (self.df_ci['level'] == ci_level)]

        if ci_type == 'profile-likelihood':
            ci_l = df_ci_last['pl_l'].item()
            ci_u = df_ci_last['pl_u'].item()
        elif ci_type == 'wald':
            ci_l = df_ci_last['wald_l'].item()
            ci_u = df_ci_last['wald_u'].item()
        elif ci_type == 'bootstrap':
            ci_l = df_ci_last['boot_l'].item()
            ci_u = df_ci_last['boot_u'].item()

        g.ax.axvspan(ci_l, ci_u, alpha=0.1, color='#7b7b7b')  # alpha is trans
        if ci_level is None:
            ci_level = self.ci_level
        label_ci = ('Confidence ({0:.2f})'.format(ci_level))

        if self.eonr <= self.df_data[self.col_n_app].max():
            alpha_axvline = 1
            g.ax.axvline(ci_l, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=label_ci, alpha=alpha_axvline)
            g.ax.axvline(ci_u, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=None, alpha=alpha_axvline)
            g.ax.axvline(self.eonr,
                         linestyle='--',
                         linewidth=1.5,
                         color='#555555',
                         label='EONR',
                         alpha=alpha_axvline)
        else:
            alpha_axvline = 0
            g.ax.axvline(0, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=label_ci, alpha=alpha_axvline)
            g.ax.axvline(0, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=None, alpha=alpha_axvline)
            g.ax.axvline(0,
                         linestyle='--',
                         linewidth=1.5,
                         color='#555555',
                         label='EONR',
                         alpha=alpha_axvline)
        return g

    def _cmap_to_palette(self, cmap):
        '''
        Converts matplotlib cmap to seaborn color palette
        '''
        palette = []
        for row in cmap:
            palette.append(tuple(row))
        return palette

    def _color_to_palette(self, color='b', cmap_len=3):
        '''
        Converts generic color into a Seaborn color palette
        <color> can be a string abbreviation for a given color (e.g.,
        'b' = blue);
        <color> can also be a colormap as an array (e.g.,
        plt.get_cmap('viridis', 3))
        '''
        try:
            color_array = colors.to_rgba_array(color)
            palette = self._cmap_to_palette(color_array)
        except ValueError:  # convert to colormap
            color_array = plt.get_cmap(color, cmap_len)
            palette = self._cmap_to_palette(color_array.colors)
        except TypeError:  # convert to palette
            palette = self._cmap_to_palette(colors)
        return palette

    def _draw_lines(self, g, palette):
        '''
        Draws N cost on plot
        '''
        g.ax.plot(self.df_linspace['x'],
                  self.df_linspace['quad_plat'],
#                  self.linspace_qp[0],
#                  self.linspace_qp[1],
                  color=palette[0],
                  linestyle='-.',
                  label='Gross return to N')
        if self.cost_n_social > 0:
            g.ax.plot(self.df_linspace['x'],
                      self.df_linspace['rtn'],
#                      self.linspace_rtn[0],
#                      self.linspace_rtn[1],
                      color=palette[2],
                      linestyle='-',
                      label='Net return to N')
            # Social cost of N is net NUP * SCN (NUP - inorganic N)
            pal2 = self._seaborn_palette(color='hls', cmap_len=10)[1]
            g.ax.plot(self.df_linspace['x'],
                      self.df_linspace['cost_n_social'],
#                      self.linspace_cost_n_social[0],
#                      self.linspace_cost_n_social[1],
                      color=pal2,
                      linestyle='-',
                      linewidth=1,
                      label='Social cost from exc. N')
            total = self.df_linspace['cost_n_fert'] +\
                self.df_linspace['cost_n_social']
#            total = self.linspace_cost_n_fert[1] +\
#                self.linspace_cost_n_social[1]
            g.ax.plot(self.df_linspace['x'],
#                      self.linspace_cost_n_social[0],
                      total,
                      color=palette[3],
                      linestyle='--',
                      label='Total N cost')
        elif self.cost_n_fert != 0:
            g.ax.plot(self.df_linspace['x'],
                      self.df_linspace['rtn'],
#                      self.linspace_rtn[0],
#                      self.linspace_rtn[1],
                      color=palette[2],
                      linestyle='-',
                      label='Net return to N')
            g.ax.plot(self.df_linspace['x'],
                      self.df_linspace['cost_n_fert'],
#                      self.linspace_cost_n_fert[0],
#                      self.linspace_cost_n_fert[1],
                      color=palette[3],
                      linestyle='--',
                      label='N fertilizer cost')
        return g

    def _modify_axes_labels(self, fontsize=14):
        plt.ylabel('Return to N ({0})'.format(self.unit_rtn),
                   fontweight='bold',
                   fontsize=fontsize)
        xlab_obj = plt.xlabel('N Rate ({0})'.format(self.unit_nrate),
                              fontweight='bold',
                              fontsize=fontsize)
        plt.getp(xlab_obj, 'color')

    def _modify_axes_pos(self, g):
        '''
        Modifies axes positions
        '''
        box = g.ax.get_position()
        g.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return g

    def _modify_minmax_axes(self, g, x_min=None, x_max=None, y_min=None,
                            y_max=None):
        '''
        Attempts to modify the min and max axis values
        <x_min> and <x_max> OR <y_min> and <y_max> must be provided, otherwise
        nothing will be set
        '''
        try:
            x_min
            x_max
            x_true = True
        except NameError:
            x_true = False
        try:
            y_min
            y_max
            y_true = True
        except NameError:
            y_true = False
        if x_true is True:
            g.fig.axes[0].set_xlim(x_min, x_max)
        if y_true is True:
            g.fig.axes[0].set_ylim(y_min, y_max)
        return g

    def _modify_plot_params(self, g, plotsize_x=7, plotsize_y=4, labelsize=11):
        '''
        Modifies plot size, changes font to bold, and sets tick/label size
        '''
        fig = plt.gcf()
        fig.set_size_inches(plotsize_x, plotsize_y)
        plt.rcParams['font.weight'] = 'bold'
        g.ax.xaxis.set_tick_params(labelsize=labelsize)
        g.ax.yaxis.set_tick_params(labelsize=labelsize)
        g.fig.subplots_adjust(top=0.9, bottom=0.135, left=0.11, right=0.99)
        return g

    def _place_legend(self, g, loc='upper left', facecolor='white'):
        '''
        2: GRTN
        3: NRTN
        4: Cost N fert
        5: Cost N social
        0: Confidence intervals
        '''
        h, leg = g.ax.get_legend_handles_labels()
        if self.cost_n_social > 0:
            order = [2, 3, 4, 5, 0]
            handles = [h[idx] for idx in order]
            labels = [leg[idx] for idx in order]
        elif self.cost_n_fert != 0:
            order = [2, 3, 4, 0]
            handles = [h[idx] for idx in order]
            labels = [leg[idx] for idx in order]
        else:
            order = [2, 0]
            handles = [h[idx] for idx in order]
            labels = [leg[idx] for idx in order]
        patch_ci = mpatches.Patch(facecolor='#7b7b7b', edgecolor='k',
                                  alpha=0.3, fill=True, linewidth=0.5)

        handles[-1] = patch_ci
        leg = g.ax.legend(handles=handles,
                          labels=labels,
                          loc=loc,
                          frameon=True,
                          framealpha=0.75,
                          facecolor='white',
                          fancybox=True,
                          borderpad=0.5,
                          edgecolor=(0.5, 0.5, 0.5),
                          prop={
                                  'weight': 'bold',
                                  'size': 7
                                  })
        for text in leg.get_texts():
            plt.setp(text, color='#444444')
        return g

    def _plot_points(self, col_x, col_y, data, palette, ax=None, s=20,
                     zorder=1):
        '''
        Plots points on figure
        <zorder> defaults to 2.6 because for some reason the background grid
        is around 2.5 (and None must set to 0 or something like that..)
        '''
        if ax is None:
            g = sns.scatterplot(x=col_x,
                                y=col_y,
                                color=palette,
                                legend=False,
                                data=data,
                                s=s,
                                zorder=zorder)
        else:
            g = sns.scatterplot(x=col_x,
                                y=col_y,
                                color=palette,
                                legend=False,
                                data=data,
                                ax=ax,
                                s=s,
                                zorder=zorder)
        return g

    def _seaborn_palette(self, color='hls', cmap_len=8):
        '''
        '''
        palette = sns.color_palette(color, cmap_len)
        return palette

    def _set_style(self, style):
        '''
        Make sure style is supported by matplotlib
        '''
        try:
            plt.style.use(style)
        except OSError as err:
            print(err)
            print('Using "ggplot" style instead..'.format(style))
            plt.style.use('ggplot')

    def _der_draw_lines(self, g, ci_level=None):
        '''
        Adds the lineplot to each axes
        '''
        ax1, ax2 = g.axes[0], g.axes[1]
        if ci_level is None:
            ci_level = self.ci_level
        last_run_n = self.df_ci['run_n'].max()
        df_ci_last = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
                                (self.df_ci['level'] == ci_level)]
        pl_l = df_ci_last[df_ci_last['level'] == ci_level]['pl_l'].item()
        pl_u = df_ci_last[df_ci_last['level'] == ci_level]['pl_u'].item()

        df_opt = self.df_linspace[(self.df_linspace['x'] >= pl_l) &
                                  (self.df_linspace['x'] <= pl_u)]
        der_max = self.df_linspace['rtn_der'].iloc[-10:].max()
        df_trim = self.df_linspace[(self.df_linspace['rtn_der'] <= der_max) &
                                   (self.df_linspace['rtn_der2'] > 0.001)]

        ax1.plot(df_opt['x'],
                 df_opt['rtn'],
                 color=self.palette[2],
                 linestyle='-',
                 label='Net return to N')
        ax2.plot(df_trim['x'],
                 df_trim['rtn_der'],
                 color=self.palette[0],
                 linestyle='-',
                 label=r'|$\Delta$Net return to N|')
        return g

    def _der_add_ci(self, g, ci_type='profile-likelihood', ci_level=None):
        '''
        Adds upper and lower confidence curves to the EONR plot
        <ci_type> --> type of confidence interval to display
            'profile-likelihood' (default) is the profile likelihood confidence
            interval (see Cook and Weisberg, 1990)
            'wald' is the Wald (uniform) confidence interval
            if the profile-likelihood confidence interval is available, it is
            recommended to use this for EONR data
        <ci_level> --> the confidence level to display
        '''
        msg = ('Please choose either "profile-likelihood" or "wald" for '
               '<ci_type>')
        ci_types = ['profile-likelihood', 'wald', 'bootstrap']
        assert ci_type.lower() in ci_types, msg
        if ci_level is None:
            ci_level = self.ci_level
        last_run_n = self.df_ci['run_n'].max()
        df_ci_last = self.df_ci[(self.df_ci['run_n'] == last_run_n) &
                                (self.df_ci['level'] == ci_level)]

        if ci_type == 'profile-likelihood':
            ci_l = df_ci_last['pl_l'].item()
            ci_u = df_ci_last['pl_u'].item()
        elif ci_type == 'wald':
            ci_l = df_ci_last['wald_l'].item()
            ci_u = df_ci_last['wald_u'].item()
        elif ci_type == 'bootstrap':
            ci_l = df_ci_last['boot_l'].item()
            ci_u = df_ci_last['boot_u'].item()

        g.axes[0].axvspan(ci_l, ci_u, alpha=0.1, color='#7b7b7b')  # alpha is trans

        label_ci = ('Confidence ({0:.2f})'.format(ci_level))

        if self.eonr <= self.df_data[self.col_n_app].max():
            alpha_axvline = 1
            g.axes[0].axvline(ci_l, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=label_ci, alpha=alpha_axvline)
            g.axes[0].axvline(ci_u, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=None, alpha=alpha_axvline)
            g.axes[0].axvline(self.eonr,
                         linestyle='--',
                         linewidth=1.5,
                         color='#555555',
                         label='EONR',
                         alpha=alpha_axvline)
        else:
            alpha_axvline = 0
            g.axes[0].axvline(0, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=label_ci, alpha=alpha_axvline)
            g.axes[0].axvline(0, linestyle='-', linewidth=0.5, color='#7b7b7b',
                         label=None, alpha=alpha_axvline)
            g.axes[0].axvline(0,
                         linestyle='--',
                         linewidth=1.5,
                         color='#555555',
                         label='EONR',
                         alpha=alpha_axvline)
        return g

    def _der_add_labels(self, g, x_max=None, fert_range=False):
        '''
        Addes labels to the derivative plot
        '''
        if x_max is None:
            _, x_max = g.axes[1].get_xlim()
        df_ci = self.df_ci[self.df_ci['run_n'] ==
                           self.df_ci['run_n'].max()]
        boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
        el = mpatches.Ellipse((0, 0), 0.3, 0.3, angle=50, alpha=0.5)
        g.axes[0].add_artist(el)

        label_eonr = '{0}: {1:.0f} {2}\nMRTN: ${3:.2f}'.format(
                self.onr_acr, self.eonr, self.unit_nrate, self.mrtn)
        if self.eonr <= x_max:
            if fert_range is True:
                err_l = [self.eonr -
                         df_ci[df_ci['level'] == 0.5]['pl_l'].item()]
                err_u = [df_ci[df_ci['level'] == 0.5]['pl_u'].item() -
                         self.eonr]
                g.axes[0].errorbar(x=self.eonr, y=self.mrtn,
                                  xerr=[err_l, err_u], marker='.',
                                  markersize=15, color=self.palette[2],
                                  markeredgecolor='white', ecolor='#7b7b7b',
                                  capsize=6, capthick=1.5, label=label_eonr)
            else:
                g.axes[0].plot([self.eonr], [self.mrtn], marker='.',
                               markersize=15, color=self.palette[2],
                               markeredgecolor='white', label=label_eonr)
        if self.eonr > 0 and self.eonr < self.df_data[self.col_n_app].max():
            g.axes[0].annotate(
                label_eonr,
                xy=(self.eonr, self.mrtn), xytext=(-80, -30),
                textcoords='offset points', ha='left', va='top', fontsize=8,
                color='#444444',
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5)),
                arrowprops=dict(arrowstyle='-|>',
                                color="grey",
                                patchB=el,
                                shrinkB=10,
                                connectionstyle='arc3,rad=-0.3'))
        else:
            g.axes[0].annotate(
                label_eonr,
                xy=(0, 0), xycoords='axes fraction', xytext=(0.98, 0.05),
                ha='right', va='bottom', fontsize=8, color='#444444',
                bbox=dict(boxstyle=boxstyle_str, pad=0.5, fc=(1, 1, 1),
                          ec=(0.5, 0.5, 0.5)))

        label_econ = ('Grain price: ${0:.2f}\nN fertilizer cost: ${1:.2f}'
                      ''.format(self.price_grain, self.cost_n_fert))

        if self.costs_fixed != 0.0:
            label_econ = ('{0}\nFixed costs: ${1:.2f}'
                          ''.format(label_econ, self.costs_fixed))
        if self.cost_n_social > 0 and self.metric is True:
            label_econ = ('{0}\nSocial cost of N: ${1:.2f}\nPrice ratio: '
                          '{2:.2f}'.format(label_econ, self.cost_n_social,
                                           self.price_ratio))
        elif self.cost_n_social > 0 and self.metric is False:
            label_econ = ('{0}\nSocial cost of N: ${1:.2f}\nPrice ratio: '
                          '{2:.3f}'.format(label_econ, self.cost_n_social,
                                           self.price_ratio))
        elif self.cost_n_social == 0 and self.metric is True:
            label_econ = ('{0}\nPrice ratio: {1:.2f}'
                          ''.format(label_econ, self.price_ratio))
        else:
            label_econ = ('{0}\nPrice ratio: {1:.3f}'
                          ''.format(label_econ, self.price_ratio))

        if self.base_zero is True:
            label_econ = ('{0}\nBase zero: {1}{2:.2f}'.format(
                    label_econ, self.unit_currency,
                    self.coefs_grtn_primary['b0'].n))
        g.axes[1].annotate(
            label_econ,
            xy=(0, 1), xycoords='axes fraction', xytext=(0.98, 0.95),
            horizontalalignment='right', verticalalignment='top',
            fontsize=7, color='#444444',
            bbox=dict(boxstyle=boxstyle_str,
                      fc=(1, 1, 1), ec=(0.5, 0.5, 0.5), alpha=0.75))
        return g

    def _der_add_onr(self, g, alpha_lines=0.5):
        '''
        Adds vertical line where ONR is
        '''
        y_min, y_max = g.axes[0].get_ylim()
        g.axes[0].axvline(x=self.eonr, ymin=y_min, ymax=y_max, linestyle='--',
                     linewidth=1.5, color='#7b7b7b', alpha=alpha_lines)
        return g

    def _der_modify_axes_labels(self, g, fontsize=12):
        ylabel1 = 'Return to N\n({0})'.format(self.unit_rtn)
        ylabel2 = (r'|$\Delta$ Return to N|''\n''(\{0} per {1})'
                   ''.format(self.unit_rtn, self.unit_fert))
        xlabel = 'N Rate ({0})'.format(self.unit_nrate)

        g.axes[0].set_ylabel(ylabel1, fontweight='bold', fontsize=fontsize)
        g.axes[1].set_ylabel(ylabel2, fontweight='bold', fontsize=fontsize)
        g.axes[1].set_xlabel(xlabel, fontweight='bold', fontsize=fontsize)
        return g

    def _der_place_legend(self, g, loc='best', facecolor='white'):
        # Upper legend
        h1, leg1 = g.axes[0].get_legend_handles_labels()
#        order = list(range(len(h1)))
        order = [1, 2]
        handles = [h1[idx] for idx in order]
        labels = [leg1[idx] for idx in order]
        patch_ci = mpatches.Patch(facecolor='#7b7b7b', edgecolor='k',
                                  alpha=0.3, fill=True, linewidth=0.5)
        handles[-1] = patch_ci
        leg1 = g.axes[0].legend(handles=handles,
                               labels=labels,
                               loc='upper right',
                               frameon=True,
                               framealpha=0.75,
                               facecolor='white',
                               fancybox=True,
                               borderpad=0.5,
                               edgecolor=(0.5, 0.5, 0.5),
                               prop={'weight': 'bold',
                                     'size': 7})
        for text in leg1.get_texts():
            plt.setp(text, color='#444444')
        # Lower legend
        h2, leg2 = g.axes[1].get_legend_handles_labels()
        order = list(range(len(h2)))
        handles = [h2[idx] for idx in order]
        labels = [leg2[idx] for idx in order]
        leg2 = g.axes[1].legend(handles=handles,
                               labels=labels,
                               loc=loc,
                               frameon=True,
                               framealpha=0.75,
                               facecolor='white',
                               fancybox=True,
                               borderpad=0.5,
                               edgecolor=(0.5, 0.5, 0.5),
                               prop={'weight': 'bold',
                                     'size': 7})
        for text in leg2.get_texts():
            plt.setp(text, color='#444444')
        return g

    def _der_modify_plot_params(self, g, plotsize_x=7, plotsize_y=6,
                                labelsize=11):
        '''
        Modifies plot size, changes font to bold, and sets tick/label size
        '''
        fig = plt.gcf()
        fig.set_size_inches(plotsize_x, plotsize_y)
        plt.rcParams['font.weight'] = 'bold'
        g.axes[0].xaxis.set_tick_params(labelsize=labelsize)
        g.axes[0].yaxis.set_tick_params(labelsize=labelsize)
#        g.fig.subplots_adjust(top=0.9, bottom=0.135, left=0.11, right=0.99)
        return g

    def _dt_add_onr(self, g, alpha_lines=0.5):
        '''
        Adds vertical line where ONR is
        '''
        y_min, y_max = g.fig.axes[0].get_ylim()
        g.ax.axvline(x=self.eonr, ymin=y_min, ymax=y_max, linestyle='--',
                     linewidth=1.5, color='#7b7b7b', alpha=alpha_lines)
        return g

    def _dt_modify_axes_labels(self, fontsize=11):
        y_label = r'$\Delta$ $\tau$($\theta_2$)'

        plt.ylabel(y_label,
                   fontweight='bold',
                   fontsize=fontsize)
        x_label = r'$\tilde{\theta_2}$'
        xlab_obj = plt.xlabel(x_label,
                              fontweight='bold',
                              fontsize=fontsize)
        plt.getp(xlab_obj, 'color')

    def _dt_place_legend_tau(self, g, loc='best', facecolor='white'):
        h, leg = g.ax.get_legend_handles_labels()
        order = list(range(len(h)))
        handles = [h[idx] for idx in order]
        labels = [leg[idx] for idx in order]
        leg = g.ax.legend(title='CI Level',
                          handles=handles,
                          labels=labels,
                          loc=loc,
                          frameon=True,
                          framealpha=0.75,
                          facecolor='white',
                          fancybox=True,
                          borderpad=0.5,
                          edgecolor=(0.5, 0.5, 0.5),
                          prop={
                                  'weight': 'bold',
                                  'size': 9
                            })
        for text in leg.get_texts():
            plt.setp(text, color='#444444')
        return g

    def _dt_plot(self, g, df_delta_tstat, level_list, lw_thick=1.6,
                 lw_thin=1.1):
        '''
        Draws delta tau/t-stat as a function of theta2_hat
        <emphasis> --> what curve to draw with empahsis
        <df_ci> --> the dataframe containing confidence interval data
        <y_ci> --> the y data to plot
        '''
        for idx, level in enumerate(level_list):
            palette = self._seaborn_palette(color='PuBuGn_d',
                                            cmap_len=len(level_list))
            df_level = df_delta_tstat[df_delta_tstat['level'] == level]
            g.ax.plot(df_level['rate_n'], df_level['delta_tstat'],
                      color=palette[idx], linestyle='-', linewidth=lw_thick,
                      label='{0:.2f}'.format(level))

#        g.ax.plot(df_delta_tstat['rate_n'], df_delta_tstat['delta_tstat'],
#                  color=self.palette[3], linestyle='-', linewidth=lw_thick,
#                  label='$|\delta|$ $|\tau|$($|\theta2|$)/T-stat')
        return g

    def _pci_modify_axes_labels(self, y_axis, fontsize=11):
        if y_axis == 't_stat':
            y_label = r'$|\tau|$ (T statistic)'
        elif y_axis == 'f_stat':
            y_label = r'$|\tau|$ (F statistic)'
        else:
            y_label = 'Confidence (%)'
        plt.ylabel(y_label,
                   fontweight='bold',
                   fontsize=fontsize)
        xlab_obj = plt.xlabel('True {0} ({1})'.format(self.onr_acr,
                                                      self.unit_nrate),
                              fontweight='bold',
                              fontsize=fontsize)
        plt.getp(xlab_obj, 'color')

    def _pci_place_legend_tau(self, g, df_ci=None, loc='best',
                              facecolor='white'):
        h, leg = g.ax.get_legend_handles_labels()
        if df_ci is None:
            order = [0, 2, 4]
        elif not df_ci['boot_l'].isnull().all():
            order = [0, 2, 4]
        else:
            order = [0, 2]
        handles = [h[idx] for idx in order]
        labels = [leg[idx] for idx in order]
        leg = g.ax.legend(handles=handles,
                          labels=labels,
                          loc=loc,
                          frameon=True,
                          framealpha=0.75,
                          facecolor='white',
                          fancybox=True,
                          borderpad=0.5,
                          edgecolor=(0.5, 0.5, 0.5),
                          prop={
                                  'weight': 'bold',
                                  'size': 7
                            })
        for text in leg.get_texts():
            plt.setp(text, color='#444444')
        return g

    def _pci_add_ci_level(self, g, df_ci, tau_list=None, alpha_lines=0.5,
                          y_axis='t_stat', emphasis='profile-likelihood',
                          zorder=-1):
        '''
        Overlays confidence levels over the t-statistic (more intuitive)
        '''
        msg = ('Please set <y_axis> to either "t_stat", "f_stat", or "level" '
               'for plotting confidence interval curves.')
        assert y_axis in ['t_stat', 'f_stat', 'level'], msg
        if tau_list is None:
            if y_axis == 't_stat':
                tau_list = list(zip(df_ci[y_axis], df_ci['level']))
            elif y_axis == 'f_stat':
                tau_list = list(zip(df_ci[y_axis], df_ci['level']))
            else:
                tau_list = list(zip(df_ci[y_axis], df_ci['t_stat']))
        x_min, x_max = g.fig.axes[0].get_xlim()
        y_min, y_max = g.fig.axes[0].get_ylim()
        # to conform with _add_labels()
        x_max_new = ((x_max - x_min) * 1.1) + x_min  # adds 10% to right
        for idx, tau_pair in enumerate(tau_list):
            if emphasis == 'none':
                xmin_data = min(df_ci[df_ci[y_axis] ==
                                      tau_pair[0]]['wald_l'].item(),
                                df_ci[df_ci[y_axis] ==
                                      tau_pair[0]]['pl_l'].item(),
                                df_ci[df_ci[y_axis] ==
                                      tau_pair[0]]['boot_l'].item())
                x_val_l = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_l'].item()
                x_val_u = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_u'].item()
            elif emphasis == 'wald':
                xmin_data = df_ci[df_ci[y_axis] ==
                                  tau_pair[0]]['wald_l'].item()
                x_val_l = df_ci[df_ci[y_axis] == tau_pair[0]]['wald_l'].item()
                x_val_u = df_ci[df_ci[y_axis] == tau_pair[0]]['wald_u'].item()
            elif emphasis == 'profile-likelihood':
                xmin_data = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_l'].item()
                x_val_l = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_l'].item()
                x_val_u = df_ci[df_ci[y_axis] == tau_pair[0]]['pl_u'].item()
            elif emphasis == 'bootstrap':
                xmin_data = df_ci[df_ci[y_axis] ==
                                  tau_pair[0]]['boot_l'].item()
                x_val_l = df_ci[df_ci[y_axis] == tau_pair[0]]['boot_l'].item()
                x_val_u = df_ci[df_ci[y_axis] == tau_pair[0]]['boot_u'].item()
            xmin = (xmin_data - x_min) / (x_max_new - x_min)

            if idx == 0:
                xmax = (x_max - x_min) / ((x_max_new*1.05) - x_min)
            else:
                xmax = (x_max - x_min) / (x_max_new - x_min)

            if y_axis == 'level':
                y_val = tau_pair[0] * 100
                ymax = (tau_pair[0]*100 - y_min) / (y_max - y_min)
            else:
                y_val = tau_pair[0]
                ymax = (tau_pair[0] - y_min) / (y_max - y_min)

            # skips 0.7 CI
            if len(tau_list) - idx <= 6 and len(tau_list) - idx != 5:
                g.ax.axhline(y_val, xmin=xmin, xmax=xmax, linestyle='--',
                             linewidth=0.5, color='#7b7b7b',
                             label='{0:.2f}'.format(tau_pair[1]),
                             alpha=alpha_lines, zorder=zorder)
#                zorder += 1
                g.ax.axvline(x_val_l, ymax=ymax, linestyle='--', linewidth=0.5,
                             color='#7b7b7b', alpha=alpha_lines, zorder=zorder)
#                zorder += 1
                g.ax.axvline(x_val_u, ymax=ymax, linestyle='--', linewidth=0.5,
                             color='#7b7b7b', alpha=alpha_lines, zorder=zorder)
#                zorder += 1
        return g

    def _pci_plot_emphasis(self, g, emphasis, df_ci, y_ci, lw_thick=1.6,
                           lw_thin=1.1):
        '''
        Draws confidence interval curves considering emphasis
        <emphasis> --> what curve to draw with empahsis
        <df_ci> --> the dataframe containing confidence interval data
        <y_ci> --> the y data to plot
        '''
        if emphasis.lower() == 'bootstrap':
            msg = ('Bootstrap CIs were not computed and thus can not be '
                   'emphasized. Please set "bootstrap_ci=True" or change the '
                   'empahsis to "wald" or "profile-likelihood".')
            assert not df_ci['boot_l'].isnull().all(), msg
        if emphasis.lower() == 'wald':
            g.ax.plot(df_ci['wald_l'], y_ci,
                      color=self.palette[3], linestyle='-', linewidth=lw_thick,
                      label='Wald', zorder=5)
            g.ax.plot(df_ci['wald_u'], y_ci,
                      color=self.palette[3], linestyle='-', linewidth=lw_thick,
                      label='wald_u', zorder=5)
            self._plot_points(col_x='wald_l', col_y=y_ci, data=df_ci,
                              palette=self.palette[3], ax=g.ax, s=20,
                              zorder=6)
            self._plot_points(col_x='wald_u', col_y=y_ci, data=df_ci,
                              palette=self.palette[3], ax=g.ax, s=20,
                              zorder=6)
        else:
            g.ax.plot(df_ci['wald_l'], y_ci,
                      color=self.palette[3], linestyle='-', linewidth=lw_thin,
                      label='Wald', zorder=4)
            g.ax.plot(df_ci['wald_u'], y_ci,
                      color=self.palette[3], linestyle='-', linewidth=lw_thin,
                      label='wald_u', zorder=4)
        if emphasis.lower() == 'profile-likelihood':
            g.ax.plot(df_ci['pl_l'], y_ci,
                      color=self.palette[2], linestyle='--',
                      linewidth=lw_thick,
                      label='Profile-likelihood', zorder=5)
            g.ax.plot(df_ci['pl_u'], y_ci,
                      color=self.palette[2], linestyle='--',
                      linewidth=lw_thick,
                      label='profile-likelihood_u', zorder=5)
            self._plot_points(col_x='pl_l', col_y=y_ci, data=df_ci,
                              palette=self.palette[2], ax=g.ax, s=20,
                              zorder=6)
            self._plot_points(col_x='pl_u', col_y=y_ci, data=df_ci,
                              palette=self.palette[2], ax=g.ax, s=20,
                              zorder=6)
        else:
            g.ax.plot(df_ci['pl_l'], y_ci,
                      color=self.palette[2], linestyle='--', linewidth=lw_thin,
                      label='Profile Likelihood', zorder=4)
            g.ax.plot(df_ci['pl_u'], y_ci,
                      color=self.palette[2], linestyle='--', linewidth=lw_thin,
                      label='profile-likelihood_u', zorder=4)
        if emphasis.lower() == 'bootstrap':
            g.ax.plot(df_ci['boot_l'], y_ci,
                      color=self.palette[0], linestyle='-.',
                      linewidth=lw_thick,
                      label='Bootstrap', zorder=5)
            g.ax.plot(df_ci['boot_u'], y_ci,
                      color=self.palette[0], linestyle='-.',
                      linewidth=lw_thick,
                      label='bootstrap_u', zorder=5)
            self._plot_points(col_x='boot_l', col_y=y_ci, data=df_ci,
                              palette=self.palette[0], ax=g.ax, s=20,
                              zorder=6)
            self._plot_points(col_x='boot_u', col_y=y_ci, data=df_ci,
                              palette=self.palette[0], ax=g.ax, s=20,
                              zorder=6)
        elif not df_ci['boot_l'].isnull().all():
            g.ax.plot(df_ci['boot_l'], y_ci,
                      color=self.palette[0], linestyle='-.',
                      linewidth=lw_thin,
                      label='Bootstrap', zorder=4)
            g.ax.plot(df_ci['boot_u'], y_ci,
                      color=self.palette[0], linestyle='-.', linewidth=lw_thin,
                      label='bootstrap_u', zorder=4)
        return g

    def _pci_add_labels(self, g, df_ci, tau_list=None, y_axis='t_stat'):
        '''
        Adds confidence interval or test statistic labels to the plot
        '''
        msg = ('Please set <y_axis> to either "t_stat", "f_stat", or "level" '
               'for plotting confidence interval curves.')
        assert y_axis in ['t_stat', 'f_stat', 'level'], msg
        if tau_list is None:
            if y_axis == 't_stat':
                tau_list = list(zip(df_ci[y_axis], df_ci['level']))
            elif y_axis == 'f_stat':
                tau_list = list(zip(df_ci[y_axis], df_ci['level']))
            elif y_axis == 'level':
                tau_list = list(zip(df_ci[y_axis]*100, df_ci['t_stat']))
        x_min, x_max = g.fig.axes[0].get_xlim()
        y_min, y_max = g.fig.axes[0].get_ylim()
    #    x_pos = self.eonr + (x_max - self.eonr)/2
#        x_max_new = ((x_max - x_min) * 1.1) - abs(x_min)  # adds 10% to right
        x_max_new = ((x_max - x_min) * 1.1) + x_min  # adds 10% to right
        x_pos = ((x_max - x_min) * 1.07) + x_min  # adds 10% to right
        g.fig.axes[0].set_xlim(right=x_max_new)
    #    boxstyle_str = 'round, pad=0.5, rounding_size=0.15'
    #    el = mpatches.Ellipse((0, 0), 0.3, 0.3, angle=50, alpha=0.5)
    #    g.ax.add_artist(el)
        for idx, tau_pair in enumerate(tau_list):
            if y_axis == 'level':
                if idx == 0:
                    label = r'$|\tau|$ (T) = {0:.2f}'.format(tau_pair[1])
                    y_pos = tau_pair[0] + (y_max - y_min)*0.02
                else:
                    label = '{0:.2f}'.format(tau_pair[1])
            else:
                if idx == 0:
                    label = 'CI = {0:.0f}%'.format(tau_pair[1]*100)
                    y_pos = tau_pair[0] + (y_max - y_min)*0.02
                else:
                    label = '{0:.0f}%'.format(tau_pair[1]*100)
            if idx == 0:
                g.ax.annotate(label,
                              xy=(x_pos, y_pos),
                              xycoords='data',
                              fontsize=7, fontweight='light',
                              horizontalalignment='right',
                              verticalalignment='center',
                              color='#7b7b7b')
            # skips 0.7 CI
            elif len(tau_list) - idx <= 6 and len(tau_list) - idx != 5:
#            elif len(tau_list) - idx <= 5:
                g.ax.annotate(label,
                              xy=(x_pos, tau_pair[0]),
                              xycoords='data',
                              fontsize=7, fontweight='light',
                              horizontalalignment='right',
                              verticalalignment='center',
                              color='#7b7b7b')
        return g

    def _set_title_size(self, g, title_text, size_font, coef=1.3):
        '''
        Sets the font size of a text object so it fits in the figure
        1.3 works pretty well for tau plots

        '''
        text_obj_dummy = plt.text(0, 0, title_text, size=size_font)
        x_min, x_max = g.axes[0].get_xlim()
        dpi = g.get_dpi()
        width_in, _ = g.get_size_inches()
        pix_width = dpi * width_in
        r = g.canvas.get_renderer()
        pix_width = r.width
        text_size = text_obj_dummy.get_window_extent(renderer=r)
        while (text_size.width * coef) > pix_width:
            size_font -= 1
            text_obj_dummy.set_size(size_font)
            text_size = text_obj_dummy.get_window_extent(renderer=r)
        text_obj_dummy.remove()
        return size_font

    def modify_size(self, fig=None, plotsize_x=7, plotsize_y=4, labelsize=11):
        '''
        Modifies the size of the last plot generated

        Parameters:
            fig (``Matplotlib Figure``, optional): Matplotlib figure to modify
                (default: None)
            plotsize_x (``float``, optional): Sets x size of plot in inches
                (default: 7)
            plotsize_y (``float``, optional): Sets y size of plot in inches
                (default: 4)
            labelsize (``float``, optional): Sets tick and label
                (defaulat: 11)
        '''
        if fig is None:
            print('<fig> is None, so adjusting the current (most recent) '
                  'figure\n.')
            fig = plt.gcf()
        fig.set_size_inches(plotsize_x, plotsize_y)
        plt.rcParams['font.weight'] = 'bold'
        fig.axes[0].xaxis.set_tick_params(labelsize=labelsize)
        fig.axes[0].yaxis.set_tick_params(labelsize=labelsize)
        fig.tight_layout()
        return fig

    def modify_title(self, title_text, g, size_font=12):
        '''
        Parameters:
            title_text (``str``): New title text
            g (``matplotlib.figure``): Matplotlib figure object to modify
                (default: None)
            size_font (``float``): Font size to use (default: 12)
        '''
#        title_text = ('{0} {1} - {2} N Fertilizer Timing'
#                      ''.format(self.year, self.location, self.time_n))
#        msg = ('No figure available; please pass a figure using the `g` '
#               'parameter.')
#        assert self.fig_eonr is not None, msg

        try:
            ax = g.fig.axes[-1]
            g_fig = True
        except AttributeError:
            g_fig = False

        if g_fig is True:
            ax = g.fig.axes[-1]
            divider = make_axes_locatable(g.ax)
            size_font = self._set_title_size(g.fig, title_text, size_font,
                                             coef=1.15)
        else:
            ax = g.axes[-1]
            divider = make_axes_locatable(g.axes[0])
            size_font = self._set_title_size(g, title_text, size_font,
                                             coef=1.15)
        ax.set_axis_off()
        cax = divider.append_axes("top", size="15%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor('#7b7b7b')
        text_obj = AnchoredText(title_text, loc=10, frameon=False,
                                prop=dict(backgroundcolor='#7b7b7b',
                                          size=size_font, color='white',
                                          fontweight='bold'))
        text_obj.prop.set_size(size_font)
        cax.add_artist(text_obj)
        if g_fig is True:
            g.fig.show()
        else:
            g.show()
        return g

    def plot_delta_tstat(self, level_list=None, style='ggplot'):
        '''Plots the test statistic as a function nitrogen rate

        Parameters:
            level_list (``list``): The confidence levels to plot; should be a
                subset of items in EONR.ci_list (default: None).
            style (``str``, optional): The style of the plolt; can be any of
                the options supported by ``matplotlib``
        '''
        msg1 = ('"delta_tstat" was not computed. Please use the '
                'delta_tstat=True option when calculating EONR: '
                'EONR.calculate_eonr(delta_tstat=True)')
        assert self.df_delta_tstat is not None, msg1

        self._set_style(style)
        if level_list is None:
            level_list = self.df_delta_tstat['level'].unique()
        df_delta_tstat = self.df_delta_tstat.copy()
        g = sns.FacetGrid(df_delta_tstat)
        g = self._dt_add_onr(g)
        g = self._dt_plot(g, df_delta_tstat, level_list)
        self._dt_modify_axes_labels()
        g = self._dt_place_legend_tau(g)
        g = self._modify_plot_params(g, plotsize_x=4.5, plotsize_y=4,
                                     labelsize=11)
#        g.ax.tick_params(axis='both', labelsize=9)
        self._add_title(g)
#        g.ax.grid(False, axis='x')
#        y_grids = g.ax.get_ygridlines()
#        for idx, grid in enumerate(y_grids):
#            if idx == 1:
#                grid.set_linewidth(1.0)
#            else:
#                grid.set_alpha(0.4)
        plt.tight_layout()
        self.fig_delta_tstat = g

    def plot_derivative(self, ci_type='profile-likelihood', ci_level=None,
                        style='ggplot'):
        '''
        Plots a zoomed up view of the ONR and the derivative

        Parameters:
            ci_type (str): Indicates which confidence interval type should be
                plotted. Options are 'wald', to plot the Wald
                CIs; 'profile-likelihood', to plot the profile-likelihood
                CIs; or 'bootstrap', to plot the bootstrap CIs (default:
                'profile-likelihood').
            ci_level (float): The confidence interval level to be plotted, and
                must be one of the values in EONR.ci_list. If None, uses the
                EONR.ci_level (default: None).
            level (``float``): The confidence levels to plot; should be a
                value from EONR.ci_list (default: 0.90).
            style (``str``, optional): The style of the plolt; can be any of
                the options supported by ``matplotlib``
        '''
        self._set_style(style)
        g, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        g = self._der_modify_axes_labels(g)
        g = self._der_add_ci(g, ci_type=ci_type, ci_level=ci_level)
        g.axes[1].axvline(self.eonr,
                          linestyle='--',
                          linewidth=1.5,
                          color='#555555',
                          label='EONR')
        g = self._der_draw_lines(g, ci_level=0.99)
        g = self._der_add_onr(g)
#        g = self._modify_axes_pos(g)
        g = self._der_place_legend(g, loc='lower right')
        g = self._der_modify_plot_params(g, plotsize_x=6, plotsize_y=5,
                                         labelsize=11)

        g = self._der_add_labels(g)
        g = self._add_title(g, der=True)
        plt.tight_layout()

        self.fig_derivative = g

    def plot_eonr(self, ci_type='profile-likelihood', ci_level=None,
                  run_n=None, x_min=None, x_max=None, y_min=None, y_max=None,
                  show_model=True, fert_range=False, style='ggplot'):
        '''Plots EONR, MRTN, GRTN, net return, and nitrogen cost

        Parameters:
            ci_type (str): Indicates which confidence interval type should be
                plotted. Options are 'wald', to plot the Wald
                CIs; 'profile-likelihood', to plot the profile-likelihood
                CIs; or 'bootstrap', to plot the bootstrap CIs (default:
                'profile-likelihood').
            ci_level (float): The confidence interval level to be plotted, and
                must be one of the values in EONR.ci_list. If None, uses the
                EONR.ci_level (default: None).
            run_n (int): NOT IMPLEMENTED. The run number to plot, as indicated
                in EONR.df_results; if None, uses the most recent, or maximum,
                run_n in EONR.df_results (default: None).
            x_min (int): The minimum x-bounds of the plot (default: None)
            x_max (int): The maximum x-bounds of the plot (default: None)
            y_min (int): The minimum y-bounds of the plot (default: None)
            y_max (int): The maximum y-bounds of the plot (default: None)
            show_model (str): Whether to display the type of fitted model in
                the helper legend (default: True).
            fert_range (bool): Whether to include the recommended range in
                fertilizer rate; this is the 50% confidence interval (default:
                False) -- DO NOT USE.
            style (str): The style of the plolt; can be any of the options
                supported by
                supported by
                <a href="https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html">matplotlib</a>
        '''
        self._set_style(style)
        g = sns.FacetGrid(self.df_data)
        self._plot_points(self.col_n_app, 'grtn', self.df_data,
                          [self.palette[0]], ax=g.ax)
        if self.cost_n_social > 0:
            try:
                self._plot_points(self.col_n_avail, 'social_cost_n',
                                  self.df_data, [self.palette[8]], ax=g.ax)
            except KeyError as err:
                print('{0}\nFixed social cost of N was used, so points will '
                      'not be plotted..'.format(err))
        self._modify_axes_labels(fontsize=14)
        g = self._add_ci(g, ci_type=ci_type, ci_level=ci_level)
        g = self._draw_lines(g, self.palette)
        g = self._modify_minmax_axes(g, x_min=x_min, x_max=x_max,
                                     y_min=y_min, y_max=y_max)
        g = self._modify_axes_pos(g)
        g = self._place_legend(g, loc='upper left')
        g = self._modify_plot_params(g, plotsize_x=7, plotsize_y=4,
                                     labelsize=11)
        g = self._add_labels(g, x_max, show_model=show_model,
                             fert_range=fert_range)
        g = self._add_title(g)
        plt.tight_layout()

        self.fig_eonr = g

    def plot_save(self, fname=None, base_dir=None, fig=None,
                  dpi=300):
        '''Saves a generated matplotlib figure to file

        Parameters:
            fname (str): Filename to save plot to (default: None)
            base_dir (str): Base file directory when saving results (default:
                None)
            fig (Matplotlib Figure): Matplotlib figure object to save
            dpi (int): Resolution to save the figure to in dots per inch
                (default: 300)
        '''
        if fig is None:
            print('``fig`` is None, so saving the current (most recent) '
                  'figure.\n')
            fig = plt.gcf()
        msg = ('A figure must be generated first. Please execute '
               'EONR.plot_eonr() or EONR.plot_tau() first..\n')
        assert fig is not None, msg

        if fname is None:
            if (self.location is not None and self.year is not None and
                    self.time_n is not None):
                fname = 'eonr_{0}_{1}_{2}.png'.format(self.location, self.year,
                                                      self.time_n)
            elif (self.location is not None and self.year is not None):
                fname = 'eonr_{0}_{1}.png'.format(self.location, self.year)
            elif (self.location is not None):
                fname = 'eonr_{0}.png'.format(self.location)
            elif (self.year is not None):
                fname = 'eonr_{0}.png'.format(self.year)
            else:
                fname = 'eonr_figure.png'
        if base_dir is None:
            base_dir = self.base_dir
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        if os.path.dirname(fname) == '':  # passing just file, add to base_dir
            fname = os.path.join(base_dir, fname)
        else:  # passing the entire filepath, just use fname
            fname = fname
        fig.savefig(fname, dpi=dpi)

    def plot_tau(self, y_axis='t_stat', emphasis='profile-likelihood',
                 run_n=None, style='ggplot'):
        '''Plots the test statistic as a function nitrogen rate

        Parameters:
            y_axis (str): Value to plot on the y-axis. Options are 't_stat', to
                plot the *T statistic*; 'f_stat', to plot the *F-statistic*;
                or 'level', to plot the *confidence level*; (default:
                't_stat').
            emphasis (str): Indicates which confidence interval type, if any,
                should be emphasized. Options are 'wald', to empahsize the Wald
                CIs; 'profile-likelihood', to empahsize the profile-likelihood
                CIs; 'bootstrap', to empahsize the bootstrap CIs; or None, to
                empahsize no CI (default: 'profile-likelihood').
            run_n (int): The run number to plot, as indicated in EONR.df_ci; if
                None, uses the most recent, or maximum, run_n in df_ci
                (default: None).
            style (``str``, optional): The style of the plolt; can be any of
                the options supported by ``matplotlib``
        '''
        self._set_style(style)
        if run_n is None:
            df_ci = self.df_ci[self.df_ci['run_n'] ==
                               self.df_ci['run_n'].max()]
        else:
            df_ci = self.df_ci[self.df_ci['run_n'] == run_n]
        y_axis = str(y_axis).lower()
        emphasis = str(emphasis).lower()
        msg1 = ('Please set <y_axis> to either "t_stat", "f_stat", or "level" '
                'for plotting confidence interval curves.')
        assert y_axis in ['t_stat', 'f_stat', 'level'], msg1
        msg2 = ('Please set <emphasis> to either "wald", '
                '"profile-likelihood", "bootstrap", or "None" to indicate '
                'which curve (if any) should be emphasized.')
        assert emphasis in ['wald', 'profile-likelihood', 'bootstrap', 'None',
                            None], msg2
        g = sns.FacetGrid(df_ci)
        g = self._dt_add_onr(g)
        if y_axis == 'level':
            y_ci = df_ci[y_axis] * 100
        else:
            y_ci = df_ci[y_axis]
        g = self._pci_plot_emphasis(g, emphasis, df_ci, y_ci)
        self._pci_modify_axes_labels(y_axis)
        g = self._pci_place_legend_tau(g, df_ci=df_ci)
        g = self._modify_plot_params(g, plotsize_x=4.5, plotsize_y=4,
                                     labelsize=11)
        g.ax.tick_params(axis='both', labelsize=9)
        g = self._pci_add_ci_level(g, df_ci, y_axis=y_axis, emphasis=emphasis)
        g = self._pci_add_labels(g, df_ci, y_axis=y_axis)
        self._add_title(g)
        g.ax.grid(False, axis='x')
        y_grids = g.ax.get_ygridlines()
        for idx, grid in enumerate(y_grids):
            if idx == 1:
                grid.set_linewidth(1.0)
            else:
                grid.set_alpha(0.4)
        plt.tight_layout()
        self.fig_tau = g

    def update_eonr(self, EONR):
        '''Sets/updates all EONR variables required by the Plotting_tools class

        Parameters:
            EONR (EONR object): The EONR object to update
        '''
        self.df_data = EONR.df_data
        self.df_delta_tstat = EONR.df_delta_tstat
        self.cost_n_fert = EONR.cost_n_fert
        self.cost_n_social = EONR.cost_n_social
        self.costs_fixed = EONR.costs_fixed
        self.price_grain = EONR.price_grain
        self.price_ratio = EONR.price_ratio
        self.col_n_app = EONR.col_n_app
        self.col_yld = EONR.col_yld
        self.col_crop_nup = EONR.col_crop_nup
        self.col_n_avail = EONR.col_n_avail
        self.unit_currency = EONR.unit_currency
        self.unit_fert = EONR.unit_fert
        self.unit_grain = EONR.unit_grain
        self.unit_area = EONR.unit_area
        self.unit_rtn = EONR.unit_rtn
        self.unit_nrate = EONR.unit_nrate
        self.model = EONR.model
        self.model_temp = EONR.model_temp
        self.ci_level = EONR.ci_level
        self.base_dir = EONR.base_dir
        self.base_zero = EONR.base_zero
        self.print_out = EONR.print_out
        self.location = EONR.location
        self.year = EONR.year
        self.time_n = EONR.time_n
        self.onr_name = EONR.onr_name
        self.onr_acr = EONR.onr_acr
        self.R = EONR.R
        self.coefs_grtn = EONR.coefs_grtn
        self.coefs_grtn_primary = EONR.coefs_grtn_primary
        self.coefs_nrtn = EONR.coefs_nrtn
        self.coefs_social = EONR.coefs_social
        self.results_temp = EONR.results_temp
        self.mrtn = EONR.mrtn
        self.eonr = EONR.eonr
        self.df_ci = EONR.df_ci
        self.df_ci_pdf = EONR.df_ci_pdf
        self.df_der = EONR.df_der
        self.df_linspace = EONR.df_linspace
        self.fig_delta_tstat = EONR.fig_eonr
        self.fig_derivative = EONR.fig_derivative
        self.fig_eonr = EONR.fig_eonr
        self.fig_tau = EONR.fig_tau
        self.ci_list = EONR.ci_list
        self.alpha_list = EONR.alpha_list
        self.df_results = EONR.df_results
        self.bootstrap_ci = EONR.bootstrap_ci
        self.metric = EONR.metric
