import logging
import os
import pickle
import sys
from itertools import product
from numbers import Number

import ipywidgets as widgets
import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from ipyfilechooser import FileChooser
from IPython.display import HTML, display
from plotly.subplots import make_subplots
from pspipe_utils import best_fits, log, misc, pspipe_list
from pspy import pspy_utils, so_dict, so_spectra
from voila.app import Voila

import psinspect

_psinspect_dict_file = "PSINSPECT_DICT_FILE"
_psinspect_product_dur = "PSINSPECT_PRODUCT_DIR"
_psinspect_theme = "PSINSPECT_THEME"
_psinspect_debug_flag = "PSINSPECT_DEBUG_FLAG"

palette = "deep"
d = so_dict.so_dict()


# Empty widget that acts as a placeholder when updating latter the content of the tab
class Empty(widgets.HTML):
    pass


class Bunch:
    def __init__(self, **kwds):
        self.update(**kwds)

    def update(self, **kwds):
        self.__dict__.update(kwds)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return self.__dict__.__str__()


# Global log widget to catch message
logger = widgets.Output()

# Banner
banner = """
<center>
<b>
<i class="fa-solid fa-magnifying-glass fa-flip fa-10x"></i>
</b>
<p>
<b><font size="+3">PS Pipeline Inspector</font></b>
</p>
<p>
{version}
</p>
<p>
<font size="+1">Select a dict file</font>
</p>
</center>
"""


# Generate HTML widget with error msg inside
def html_err_msg(err_msg):
    return widgets.HTML(
        f"""<center>
        <p><b><font size="+2" color="red">
        <i class="fa-solid fa-link-slash fa-bounce"></i> {err_msg}
        </font></b></p>
        </center>"""
    )


class App:
    """An ipywidgets and plotly application for checking PSpipe productions"""

    def initialize(self, dict_file=None, product_dir=None, debug=False):
        self.base_layout = dict(
            height=800,
            template=os.getenv(_psinspect_theme, "plotly_white"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        # This is to fix loading of mathjax that is not correctly done and make the application totally
        # bugging see https://github.com/microsoft/vscode-jupyter/issues/8131#issuecomment-1589961116
        display(
            HTML(
                '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
                + '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.2/css/fontawesome.min.css">'
            )
        )
        self.log = log.get_logger(
            debug=os.getenv(_psinspect_debug_flag) == "True" or debug,
            fmt="%(asctime)s - %(levelname)s: %(message)s",
        )

        # To avoid red background
        # https://stackoverflow.com/questions/25360714/avoid-red-background-color-for-logging-output-in-ipython-notebook
        self.log.handlers[0].stream = sys.stdout

        if dict_file:
            os.environ[_psinspect_dict_file] = os.path.realpath(dict_file)

        self.registered_callback = list()
        self.updated_tab = set()
        self.db = dict()

        # Initialize first the footer i.e. the dumper and then the loader since a dict file can be
        # passed as argument and then will need the dumper to work
        self._initialize_footer()
        self._initialize_header()
        self._initialize_app()

    def run(self):
        display(self.app)

    @logger.capture()
    def update(self):
        self.log.info(f"Loading products from {self._product_dir} directory...")
        self.tab = self.body = widgets.Tab(children=())

        if not hasattr(self, "app"):
            self._initialize_app()
        else:
            # Adding tab to app
            children = list(self.app.children)
            children[1] = self.body
            self.app.children = children

        self._update_dict()
        self._update_binning()
        self._update_passbands()
        self._update_beams()
        self._update_windows()
        self._update_maps()
        self._update_spectra()
        self._update_best_fits()
        self._update_noise_model()
        self._update_mc_spectra()

        def _update(change=None):
            if (index := self.tab.selected_index) in self.updated_tab:
                return
            self.registered_callback[index]()
            self.updated_tab.add(index)

        self.tab.observe(_update, names="selected_index")
        if self.tab.children:
            _update()
        self.log.info(f"psinspector {psinspect.__version__} succesfully loaded")

    def _initialize_footer(self):
        self.footer = widgets.Accordion(children=[logger], titles=["Logs"])

    def _initialize_header(self):
        kwargs = {}
        if filename := os.getenv(_psinspect_dict_file, ""):
            kwargs = dict(
                path=os.path.dirname(filename),
                filename=os.path.basename(filename),
                select_default=True,
            )

        file_chooser = FileChooser(
            filter_pattern="*.dict", layout=widgets.Layout(height="auto", width="auto"), **kwargs
        )
        file_chooser.dir_icon = "/"

        def _load_dict(chooser):
            d.read_from_file(chooser.selected)
            self._product_dir = chooser.selected_path
            self.footer.selected_index = 0
            self.registered_callback.clear()
            self.updated_tab.clear()
            self.db.clear()
            self.update()

        file_chooser.register_callback(_load_dict)

        self.header = file_chooser
        if filename:
            _load_dict(file_chooser)

    def _initialize_app(self):
        if not hasattr(self, "body"):
            self.body = widgets.HTML(value=banner.format(version=psinspect.__version__))
        self.app = widgets.VBox([self.header, self.body, self.footer])

    def _add_tab(self, title="", callback=None):
        if callback in self.registered_callback:
            return False

        self.tab.children += (Empty(),)
        self.tab.set_title(len(self.tab.children) - 1, title)
        self.registered_callback += [callback]
        return True

    def _update_tab(self, widget):
        children = list(self.tab.children)
        children[self.tab.selected_index] = widget
        self.tab.children = children

    def _add_db_entry(self, key, **value):
        if isinstance(key, (str, Number)):
            key = (key,)
        self.db.setdefault(key, Bunch()).update(**value)

    def _get_db_entry(self, key):
        return self.db.get((key,) if isinstance(key, (str, Number)) else key)

    @logger.capture()
    def _has_db_entry(self, key, flatten=False):
        # self.log.debug(f"key={key}")
        # self.log.debug(sum(self.db.keys(), ()))
        keys = sum(self.db.keys(), ()) if flatten else self.db.keys()
        return key in keys

    @logger.capture()
    def _update_dict(self):
        if self._add_tab(title="Dict file", callback=self._update_dict):
            return

        self.spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
        self.lmax = d["lmax"]
        self.cross_list = pspipe_list.get_spec_name_list(d, delimiter="_")
        self.survey_list = pspipe_list.get_map_set_list(d)
        color_list = sns.color_palette(palette, n_colors=len(self.cross_list)).as_hex()
        self.colors = {name: color_list[i] for i, name in enumerate(self.cross_list)}
        self.colors.update({name: self.colors["{0}x{0}".format(name)] for name in self.survey_list})
        self.log.debug(f"survey list: {self.survey_list}")
        self.log.debug(f"cross list: {self.cross_list}")

        from pygments.styles import get_all_styles

        dumper = widgets.RadioButtons(options=["print", "json", "yaml"])
        styler = widgets.Dropdown(description="Theme", options=get_all_styles(), value="default")

        dict_dump = widgets.Output()

        def _update(change=None):
            from pygments import formatters, highlight, lexers

            dict_dump.clear_output()
            formatter = formatters.Terminal256Formatter(style=styler.value)
            with dict_dump:
                fmt = dumper.value
                if fmt == "print":
                    import pprint

                    print(
                        highlight(
                            pprint.pformat(d, sort_dicts=False), lexers.PythonLexer(), formatter
                        )
                    )
                if fmt == "json":
                    import json

                    print(highlight(json.dumps(d, indent=2), lexers.JsonLexer(), formatter))
                if fmt == "yaml":
                    import yaml

                    print(highlight(yaml.dump(d), lexers.YamlLexer(), formatter))

        dumper.observe(_update, names="value")
        styler.observe(_update, names="value")
        _update()

        self._update_tab(widgets.VBox([widgets.HBox([dumper, styler]), dict_dump]))

    def directory_exists(self, dirname):
        return None if not os.path.exists(d := os.path.join(self._product_dir, dirname)) else d

    def _refresh_figure(self, change, fig):
        if change:
            tab = self.tab.children[self.tab.selected_index]
            children = list(tab.children)
            children[-1] = go.FigureWidget(fig)
            tab.children = children
        else:
            return go.FigureWidget(fig)

    @logger.capture()
    def _update_binning(self):
        if self._add_tab(title="Binning", callback=self._update_binning):
            return

        if not os.path.exists(fn := d["binning_file"]):
            self.log.debug(err_msg := "Binning information unreachable")
            self._update_tab(html_err_msg(err_msg))

        # Read binning file
        low, high, center, size = pspy_utils.read_binning_file(fn, lmax=self.lmax)

        # Get reference spectra
        if directory := self.directory_exists("best_fits"):
            ell, cmb = best_fits.cmb_dict_from_file(
                os.path.join(directory, "cmb.dat"), lmax=self.lmax, spectra=self.spectra
            )
        else:
            ell, cmb = pspy_utils.ps_from_params(d["cosmo_params"], d["type"], lmax=self.lmax)

        base_widget = widgets.HBox(
            [
                modes := widgets.SelectMultiple(
                    description="Mode",
                    options=(mode_options := ["TT", "TE", "EE", "BB"]),
                    value=mode_options,
                ),
                palettes := widgets.Dropdown(
                    description="Palette", options=list(mpl.colormaps.keys()), value="rocket"
                ),
            ]
        )

        # Compute range of spectra for later use with vertical rectangle (plotly function add_hrect
        # is to slow and does not allow hoverinfo)
        def get_range(y, fcn):
            yext = fcn(y)
            if fcn == np.min:
                return yext * (0.95 if yext > 0 else 1.05)
            if fcn == np.max:
                return yext * (1.05 if yext > 0 else 0.95)

        yranges = {
            spec: [get_range(cmb[spec], np.min), get_range(cmb[spec], np.max)]
            for spec in mode_options
        }

        def _update(change=None):
            layout = self.base_layout.copy()
            fig = make_subplots(
                rows=(nrows := len(modes.value)),
                cols=1,
                shared_xaxes=True,
                x_title="$\ell$",
                vertical_spacing=0.0,
            )
            fig.update_layout(**layout)

            color_list = sns.color_palette(palettes.value, n_colors=len(low)).as_hex()

            for i, mode in enumerate(modes.value):
                rowcol_kwargs = dict(row=i + 1, col=1)
                fig.add_scatter(
                    name=mode,
                    x=ell,
                    y=cmb[mode],
                    line_color="gray",
                    showlegend=False,
                    **rowcol_kwargs,
                )

                ymin, ymax = yranges[mode]
                fig.update_yaxes(
                    title_text="$D^{\mathrm{%s}}_\ell\;[\mu\mathrm{K}^2]$" % mode,
                    range=[ymin, ymax],
                    **rowcol_kwargs,
                )
                for j, (x0, x1) in enumerate(zip(low, high)):
                    fig.add_scatter(
                        name="",
                        text=f"bin range [{x0}; {x1}]<br>"
                        + f"bin center {center[j]}<br>"
                        + f"bin size {size[j]}",
                        hovertemplate="%{text}",
                        x=[x0, x1, x1, x0],
                        y=[ymin, ymin, ymax, ymax],
                        mode="none",
                        fill="toself",
                        fillcolor=color_list[j],
                        opacity=0.2,
                        showlegend=False,
                        **rowcol_kwargs,
                    )

            return self._refresh_figure(change, fig)

        modes.observe(_update, names="value")
        palettes.observe(_update, names="value")

        self._update_tab(widgets.VBox([base_widget, _update()]))

    @logger.capture()
    def _update_passbands(self):
        if self._add_tab(title="Bandpass", callback=self._update_passbands):
            return

        _key = "passband"
        for survey in self.survey_list:
            nu_ghz, pb = None, None
            freq_info = d[f"freq_info_{survey}"]
            if d["do_bandpass_integration"]:
                if os.path.exists(filename := freq_info["passband"]):
                    nu_ghz, pb = np.loadtxt(filename).T
            else:
                nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.0])
            if nu_ghz is not None:
                self._add_db_entry(key=(survey, _key), nu_ghz=nu_ghz, pb=pb)

        if not self._has_db_entry(_key, flatten=True):
            self.log.error(err_msg := "Passbands information unreachable")
            self._update_tab(html_err_msg(err_msg))
            return

        layout = self.base_layout.copy()
        layout.update(dict(xaxis_title="frequency [GHz]", yaxis_title="transmission"))
        fig = go.FigureWidget(layout=layout)
        for survey in self.survey_list:
            if meta := self._get_db_entry((survey, _key)):
                fig.add_scatter(
                    name=survey, x=meta.nu_ghz, y=meta.pb, line_color=self.colors[survey]
                )

        self._update_tab(fig)

    @logger.capture()
    def _update_beams(self, fetch_data=False):
        if self._add_tab(title="Beams", callback=self._update_beams):
            return

        _key = "beam"
        for survey in self.survey_list:
            if not os.path.exists(fn_beam_T := d[f"beam_T_{survey}"]) or not os.path.exists(
                fn_beam_pol := d[f"beam_pol_{survey}"]
            ):
                continue
            ell, bl = misc.read_beams(fn_beam_T, fn_beam_pol, lmax=self.lmax)
            idx = np.where((ell >= 2) & (ell < self.lmax))
            self._add_db_entry(key=(survey, _key), ell=ell, bl=bl, idx=idx)

        if not self._has_db_entry(_key, flatten=True):
            self.log.error(err_msg := "Beams information unreachable")
            self._update_tab(html_err_msg(err_msg))
            return

        base_widget = widgets.HBox(
            [
                surveys := widgets.SelectMultiple(
                    description="Survey", options=self.survey_list, value=self.survey_list
                ),
                modes := widgets.SelectMultiple(
                    description="Mode",
                    options=(mode_options := ["T", "E", "B"]),
                    value=["T"],
                ),
            ]
        )

        def _update(change=None):
            layout = self.base_layout.copy()
            layout.update(dict(xaxis_title="$\ell$", yaxis_title="normalized beam"))
            fig = go.Figure(layout=layout)
            for survey, mode in product(surveys.value, modes.value):
                if meta := self._get_db_entry((survey, "beam")):
                    fig.add_scatter(
                        name=f"{survey} - {mode}",
                        x=meta.ell[meta.idx],
                        y=meta.bl[mode][meta.idx],
                        line_color=self.colors[survey],
                        opacity=1 - mode_options.index(mode) / len(mode_options),
                    )
            return self._refresh_figure(change, fig)

        surveys.observe(_update, names="value")
        modes.observe(_update, names="value")

        if not fetch_data:
            self._update_tab(widgets.VBox([base_widget, _update()]))

    @logger.capture()
    def _update_windows(self):
        if not (directory := self.directory_exists("windows")):
            self.log.debug("No windows directory")
            return

        if self._add_tab(title="Window masks", callback=self._update_windows):
            return

        base_widget = widgets.HBox(
            [
                surveys := widgets.SelectMultiple(
                    description="Survey", options=self.survey_list, value=[self.survey_list[0]]
                ),
                masks := widgets.SelectMultiple(
                    description="Mask",
                    options=(
                        mask_names := ["baseline", "kspace", "xlink", "baseline_ivar", "xlink_ivar"]
                    ),
                    value=["baseline"],
                ),
            ]
        )

        img_widgets = widgets.VBox()
        png_files = {
            p: os.path.join(directory, "{}_mask_{}.png".format(*p))
            for p in product(mask_names, self.survey_list)
        }

        def _update(change=None):
            img_widgets.children = []
            for survey, mask in product(surveys.value, masks.value):
                with open(png_files[mask, survey], "rb") as img:
                    img_widgets.children += (
                        widgets.HTML(f"<h2>{survey} - {mask}</h2>"),
                        widgets.Image(value=img.read(), format="png", width="auto", height=400),
                    )

        surveys.observe(_update, names="value")
        masks.observe(_update, names="value")
        _update()

        self._update_tab(widgets.VBox([base_widget, img_widgets]))
        self.log.info(f"Directory '{directory}' loaded")

    @logger.capture()
    def _update_maps(self):
        if not (directory := self.directory_exists("plots/maps")):
            self.log.debug("No plots/maps directory")
            return

        if self._add_tab(title="Maps", callback=self._update_maps):
            return

        nbr_splits = {len(d[f"maps_{survey}"]) for survey in self.survey_list}
        base_widget = widgets.HBox(
            [
                surveys := widgets.SelectMultiple(
                    description="Survey", options=self.survey_list, value=[self.survey_list[0]]
                ),
                kinds := widgets.SelectMultiple(
                    description="Type",
                    options=(kind_options := ["split", "windowed_split", "no_filter_split"]),
                    value=["split"],
                ),
                modes := widgets.SelectMultiple(
                    description="Mode",
                    options=(mode_options := ["T", "Q", "U"]),
                    value=["T"],
                ),
                splits := widgets.RadioButtons(
                    description="Split",
                    options=(split_options := range(max(nbr_splits))),
                ),
            ]
        )

        img_widgets = widgets.VBox()
        png_files = {
            p: os.path.join(directory, "{}_{}_{}_{}.png".format(*p))
            for p in product(kind_options, self.survey_list, split_options, mode_options)
        }

        def _update(change=None):
            img_widgets.children = []
            for kind, survey, mode in product(kinds.value, surveys.value, modes.value):
                if not os.path.exists(filename := png_files[kind, survey, splits.value, mode]):
                    self.log.debug(f"{filename} does not exist")
                    continue
                with open(filename, "rb") as img:
                    img_widgets.children += (
                        widgets.HTML(f"<h2>{kind} - {survey} - split {splits.value} - {mode}</h2>"),
                        widgets.Image(value=img.read(), format="png", width="auto", height=400),
                    )

        surveys.observe(_update, names="value")
        kinds.observe(_update, names="value")
        splits.observe(_update, names="value")
        modes.observe(_update, names="value")
        _update()

        self._update_tab(widgets.VBox([base_widget, img_widgets]))
        self.log.info(f"Directory '{directory}' loaded")

    @logger.capture()
    def _update_spectra(self):
        if not (directory := self.directory_exists("spectra")):
            self.log.debug("No spectra directory")
            return

        if self._add_tab(title="Spectra", callback=self._update_spectra):
            return

        for name in self.cross_list:
            cl_type = d["type"]
            for kind in (kinds := ["cross", "auto", "noise"]):
                ell, spectra = so_spectra.read_ps(
                    os.path.join(directory, f"{cl_type}_{name}_{kind}.dat"), spectra=self.spectra
                )
                self._add_db_entry(key=(name, kind), ell=ell, spectra=spectra)

            n_splits = d[f"n_splits_{name.split('_')[0]}"]
            for split in (
                splits := ["{}{}".format(*s) for s in product(range(n_splits), repeat=2)]
            ):
                if os.path.exists(fn := os.path.join(directory, f"{cl_type}_{name}_{split}.dat")):
                    ell, spectra = so_spectra.read_ps(fn, spectra=self.spectra)
                    self._add_db_entry(key=(name, split), ell=ell, spectra=spectra)

        base_widget = widgets.HBox(
            [
                spectrum := widgets.Dropdown(
                    description="Spectrum", options=self.spectra, value=self.spectra[0]
                ),
                crosses := widgets.SelectMultiple(
                    description="Cross", options=self.cross_list, value=self.cross_list
                ),
                kinds := widgets.SelectMultiple(description="Kind", options=kinds, value=["cross"]),
                splits := widgets.SelectMultiple(description="Split", options=splits),
            ]
        )

        layout = self.base_layout.copy()
        layout.update(dict(height=1000))

        @logger.capture()
        def _update(change=None):
            mode = spectrum.value
            surveys = set(sum([cross.split("x") for cross in crosses.value], []))
            nsurvey = min(len(surveys), len(crosses.value))
            subplot_titles = np.full((nsurvey, nsurvey), None)
            indices = np.triu_indices(nsurvey)[::-1]
            for i, name in enumerate(crosses.value):
                irow, icol = indices[0][i], indices[1][i]
                subplot_titles[irow, icol] = name
                if mode[0] != mode[1] and nsurvey > 1:
                    subplot_titles[icol, irow] = "{1}x{0}".format(*name.split("x"))

            fig = make_subplots(
                rows=nsurvey,
                cols=nsurvey,
                shared_xaxes="all",
                shared_yaxes="all",
                subplot_titles=subplot_titles.flatten().tolist(),
                x_title="$\ell$",
                y_title="$D_\ell\;[\mu\mathrm{K}^2]$",
                vertical_spacing=0.15 / nsurvey,
                horizontal_spacing=0.05 / nsurvey,
            )
            fig.update_layout(**layout)

            for i, name in enumerate(crosses.value):
                name1, name2 = name.split("x")
                rowcol_kwargs = dict(row=indices[0][i] + 1, col=indices[1][i] + 1)
                inverse_rowcol_kwargs = dict(row=indices[1][i] + 1, col=indices[0][i] + 1)
                fill_upper = mode[0] != mode[1] and name1 != name2 and nsurvey > 1
                line_dash = {"cross": "solid", "auto": "dash", "noise": "dot"}
                for kind in kinds.value:
                    kind_kwargs = dict(
                        name=kind,
                        line_color=self.colors[name],
                        line_dash=line_dash[kind],
                        legendgroup=kind,
                    )

                    if meta := self._get_db_entry(key=(name, kind)):
                        fig.add_scatter(
                            x=meta.ell,
                            y=meta.spectra[mode],
                            showlegend=i == 0,
                            **kind_kwargs,
                            **rowcol_kwargs,
                        )
                        if fill_upper:
                            fig.add_scatter(
                                x=meta.ell,
                                y=meta.spectra[mode[::-1]],
                                showlegend=False,
                                **kind_kwargs,
                                **inverse_rowcol_kwargs,
                            )

                for split in splits.value:
                    split_kwargs = dict(
                        name=f"split {split}",
                        line_color=self.colors[name],
                        opacity=0.25,
                        showlegend=False,
                    )

                    if meta := self._get_db_entry(key=(name, split)):
                        fig.add_scatter(
                            x=meta.ell, y=meta.spectra[mode], **split_kwargs, **rowcol_kwargs
                        )
                        if fill_upper:
                            fig.add_scatter(
                                x=meta.ell,
                                y=meta.spectra[mode[::-1]],
                                **split_kwargs,
                                **inverse_rowcol_kwargs,
                            )

                yaxes_kwargs = dict(
                    type="log" if mode == "TT" else "linear",
                    range=[-1, 4] if mode == "TT" else None,
                )

                fig.update_yaxes(**yaxes_kwargs, **rowcol_kwargs)
                if fill_upper:
                    fig.update_yaxes(**yaxes_kwargs, **inverse_rowcol_kwargs)

            return self._refresh_figure(change, fig)

        spectrum.observe(_update, names="value")
        crosses.observe(_update, names="value")
        kinds.observe(_update, names="value")
        splits.observe(_update, names="value")

        self._update_tab(widgets.VBox([base_widget, _update()]))
        self.log.info(f"Directory '{directory}' loaded")

    @logger.capture()
    def _update_mc_spectra(self):
        if not (directory := self.directory_exists("montecarlo")):
            self.log.debug("No montecarlo directory")
            return

        if self._add_tab(title="MC Spectra", callback=self._update_mc_spectra):
            return

        _key = "mc spectra"
        if not self._has_db_entry("beam", flatten=True):
            self._update_beams(fetch_data=True)
        if not self._has_db_entry("noise", flatten=True):
            self._update_noise_model(fetch_data=True)
        if not self._has_db_entry("cmb_and_fg_dict", flatten=True):
            self._update_best_fits(fetch_data=True)
        for name, kind in product(self.cross_list, (kinds := ["cross", "auto", "noise"])):
            spectra = {}
            for spec in self.spectra:
                ell, mean, std = np.loadtxt(
                    os.path.join(directory, f"spectra_{spec}_{name}_{kind}.dat"), unpack=True
                )
                spectra[spec] = Bunch(mean=mean, std=std)
            self._add_db_entry(key=(name, kind, _key), ell=ell, spectra=spectra)

            if (bestfit_dir := self.directory_exists("best_fits")) and os.path.exists(
                fn := os.path.join(bestfit_dir, f"model_{name}_{kind}.dat")
            ):
                ell, spectra = so_spectra.read_ps(fn, spectra=self.spectra)
                self._add_db_entry(key=(name, kind, "binned model"), ell=ell, spectra=spectra)

            meta = self._get_db_entry(key=(name, "cmb_and_fg"))
            if kind == "cross":
                self._add_db_entry(key=(name, kind, "model"), ell=meta.ell, spectra=meta.cmb_and_fg)

            if kind in ["noise", "auto"]:
                name1, name2 = name.split("x")
                bl1 = self._get_db_entry(key=(name1, "beam"))
                bl2 = self._get_db_entry(key=(name2, "beam"))

                sv1, sv2 = name1.split("_")[0], name2.split("_")[0]
                if sv1 == sv2:
                    ar1 = name1.replace(f"{sv1}_", "")
                    ar2 = name2.replace(f"{sv2}_", "")
                    lb, nlth = so_spectra.read_ps(
                        os.path.join(
                            self._product_dir, "noise_model", f"mean_{ar1}x{ar2}_{sv1}_noise.dat"
                        ),
                        spectra=self.spectra,
                    )
                    # nlth = self._get_db_entry(key=(name, "noise", "interpolate")).nl
                    for spec in self.spectra:
                        X, Y = spec
                        # nlth[spec] /= bl1.bl[X][bl1.idx] * bl2.bl[Y][bl2.idx]
                        nlth[spec] /= bl1.bl[X] * bl2.bl[Y]
                else:
                    nlth = {spec: np.zeros(self.lmax) for spec in self.spectra}
                if kind == "noise":
                    self._add_db_entry(key=(name, kind, "model"), ell=meta.ell, spectra=nlth)
                # if kind == "auto":
                #     spectra = {
                #         spec: meta.cmb_and_fg[spec] + nlth[spec] * d[f"n_splits_{sv1}"]
                #         for spec in self.spectra
                #     }
                #     self._add_db_entry(key=(name, kind, "model"), ell=meta.ell, spectra=spectra)

        base_widget = widgets.HBox(
            [
                plots := widgets.ToggleButtons(
                    # description="Plot",
                    options=["spectra", "residuals", "σ residuals"],
                    button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
                    tooltips=[
                        "Plot absolute spectra",
                        "Plot residuals",
                        "Plot residuals in units of σ",
                    ],
                ),
                spectrum := widgets.Dropdown(
                    description="Spectrum", options=self.spectra, value=self.spectra[0]
                ),
                crosses := widgets.SelectMultiple(
                    description="Cross", options=self.cross_list, value=self.cross_list
                ),
                kinds := widgets.SelectMultiple(description="Kind", options=kinds, value=["cross"]),
            ]
        )

        @logger.capture()
        def _update(change=None):
            mode = spectrum.value
            surveys = set(sum([cross.split("x") for cross in crosses.value], []))
            nsurvey = min(len(surveys), len(crosses.value))
            subplot_titles = np.full((nsurvey, nsurvey), None)
            indices = np.triu_indices(nsurvey)[::-1]
            for i, name in enumerate(crosses.value):
                irow, icol = indices[0][i], indices[1][i]
                subplot_titles[irow, icol] = name
                if mode[0] != mode[1] and nsurvey > 1:
                    subplot_titles[icol, irow] = "{1}x{0}".format(*name.split("x"))

            nsims = d["iStop"] - d["iStart"] + 1
            self.log.debug(f"nsims={nsims}")

            # ell factors for residuals
            ell_fac = {spec: -0.8 for spec in self.spectra}
            ell_fac.update({"TT": 1.0, "TE": 0.0, "ET": 0.0})

            y_title = r"D_\ell\;[\mu\mathrm{K}^2]"
            if plots.value == "residuals":
                if ell_fac[mode] == 1.0:
                    y_title = r"\ell\Delta " + y_title
                elif ell_fac[mode] != 0.0:
                    y_title = rf"\ell^{{{ell_fac[mode]:.1f}}}" + y_title
            if plots.value == "σ residuals":
                y_title = "\Delta D_\ell\;[\sigma]"

            fig = make_subplots(
                rows=nsurvey,
                cols=nsurvey,
                shared_xaxes="all",
                shared_yaxes="all",
                subplot_titles=subplot_titles.flatten().tolist(),
                x_title="$\ell$",
                y_title=f"${y_title}$",
                vertical_spacing=0.15 / nsurvey,
                horizontal_spacing=0.05 / nsurvey,
            )
            layout = self.base_layout.copy()
            layout.update(dict(height=1000))
            fig.update_layout(**layout)

            for i, name in enumerate(crosses.value):
                name1, name2 = name.split("x")
                rowcol_kwargs = dict(row=indices[0][i] + 1, col=indices[1][i] + 1)
                inverse_rowcol_kwargs = dict(row=indices[1][i] + 1, col=indices[0][i] + 1)
                fill_upper = mode[0] != mode[1] and name1 != name2 and nsurvey > 1
                marker_symbol = {"cross": "circle", "auto": "square", "noise": "diamond"}
                for kind in kinds.value:
                    kind_kwargs = dict(
                        name=kind,
                        mode="markers",
                        marker_color=self.colors[name],
                        marker_symbol=marker_symbol[kind],
                        legendgroup=kind,
                    )

                    meta_sim = self._get_db_entry(key=(name, kind, _key))
                    meta_model = self._get_db_entry(key=(name, kind, "model"))
                    meta_binned_model = self._get_db_entry(key=(name, kind, "binned model"))

                    if plots.value == "spectra":
                        ell = meta_sim.ell
                        get_sim_values = lambda mode: meta_sim.spectra[mode].mean
                        get_sim_errors = lambda mode: meta_sim.spectra[mode].std
                    if plots.value == "residuals":
                        ell = meta_sim.ell
                        get_sim_values = (
                            lambda mode: (
                                meta_sim.spectra[mode].mean - meta_binned_model.spectra[mode]
                            )
                            * ell ** ell_fac[mode]
                        )
                        get_sim_errors = (
                            lambda mode: meta_sim.spectra[mode].std
                            / np.sqrt(nsims)
                            * ell ** ell_fac[mode]
                        )
                    if plots.value == "σ residuals":
                        ell = meta_sim.ell
                        get_sim_values = lambda mode: (
                            (meta_sim.spectra[mode].mean - meta_binned_model.spectra[mode])
                            / meta_sim.spectra[mode].std
                            / np.sqrt(nsims)
                        )
                        get_sim_errors = lambda mode: None

                    # Simulation
                    get_kwargs = lambda mode, showlegend: dict(
                        x=ell,
                        y=get_sim_values(mode),
                        error_y=dict(type="data", array=get_sim_errors(mode)),
                        showlegend=showlegend,
                        **kind_kwargs,
                    )

                    fig.add_scatter(**get_kwargs(mode, i == 0), **rowcol_kwargs)
                    if fill_upper:
                        fig.add_scatter(**get_kwargs(mode[::-1], False), **inverse_rowcol_kwargs)

                    # Model
                    if plots.value == "spectra":
                        model_kwargs = dict(
                            name="model",
                            x=meta_model.ell,
                            y=meta_model.spectra[mode],
                            line_color="gray",
                            showlegend=i == 0,
                            legendgroup="model",
                            legendrank=1,
                        )
                        fig.add_scatter(**model_kwargs, **rowcol_kwargs)
                        if fill_upper:
                            fig.add_scatter(**model_kwargs, **inverse_rowcol_kwargs)

                yaxes_kwargs = dict(type="linear")
                if plots.value == "spectra":
                    yaxes_kwargs = dict(type="log" if mode == "TT" else "linear")
                fig.update_yaxes(**yaxes_kwargs, **rowcol_kwargs)
                if fill_upper:
                    fig.update_yaxes(**yaxes_kwargs, **inverse_rowcol_kwargs)

            return self._refresh_figure(change, fig)

        plots.observe(_update, names="value")
        spectrum.observe(_update, names="value")
        crosses.observe(_update, names="value")
        kinds.observe(_update, names="value")

        self._update_tab(widgets.VBox([base_widget, _update()]))
        self.log.info(f"Directory '{directory}' loaded")

    @logger.capture()
    def _update_best_fits(self, fetch_data=False):
        if not (directory := self.directory_exists("best_fits")):
            self.log.debug("No best fits directory")
            return

        if self._add_tab(title="CMB & Foregrounds", callback=self._update_best_fits):
            return

        for name in self.cross_list:
            ell, cmb_and_fg_dict = best_fits.cmb_dict_from_file(
                os.path.join(directory, f"cmb_and_fg_{name}.dat"),
                lmax=self.lmax,
                spectra=self.spectra,
            )
            self._add_db_entry(key=(name, "cmb_and_fg"), ell=ell, cmb_and_fg=cmb_and_fg_dict)

        # Individual foregrounds
        if os.path.exists(fn := os.path.join(directory, "foregrounds.pkl")):
            with open(fn, "rb") as f:
                self._add_db_entry(key="foregrounds", **pickle.load(f))

        # Theoritical CMB
        ell, cmb = best_fits.cmb_dict_from_file(
            os.path.join(directory, "cmb.dat"),
            lmax=self.lmax,
            spectra=self.spectra,
        )
        self._add_db_entry(key="cmb", ell=ell, cmb=cmb)

        # Set foregrounds name and unique color
        fg_components = d["fg_components"]
        for mode in fg_components:
            if (comp := "tSZ_and_CIB") in fg_components[mode]:
                fg_components[mode].remove(comp)
                fg_components[mode] += ["tSZ", "cibc", "tSZxCIB"]
        fg_colors = sorted(set(sum(fg_components.values(), [])))
        color_list = sns.color_palette(palette, n_colors=len(fg_colors)).as_hex()
        fg_colors = {comp: color_list[i] for i, comp in enumerate(fg_colors)}

        base_widget = widgets.HBox(
            [
                spectrum := widgets.Dropdown(
                    options=(spectra := ["TT", "TE", "TB", "EE", "EB", "BB"]),
                    value=spectra[0],
                    description="Spectrum",
                ),
                crosses := widgets.SelectMultiple(
                    description="Cross", options=self.cross_list, value=self.cross_list
                ),
            ]
        )

        @logger.capture()
        def _update(change=None):
            mode = spectrum.value
            surveys = set(sum([cross.split("x") for cross in crosses.value], []))
            nsurvey = min(len(surveys), len(crosses.value))
            subplot_titles = np.full((nsurvey, nsurvey), None)
            indices = np.triu_indices(nsurvey)[::-1]
            for i, name in enumerate(crosses.value):
                irow, icol = indices[0][i], indices[1][i]
                subplot_titles[irow, icol] = name
                if mode[0] != mode[1] and nsurvey > 1:
                    subplot_titles[icol, irow] = "{1}x{0}".format(*name.split("x"))

            fig = make_subplots(
                rows=nsurvey,
                cols=nsurvey,
                shared_xaxes="all",
                shared_yaxes="all",
                subplot_titles=subplot_titles.flatten().tolist(),
                x_title="$\ell$",
                y_title="$D_\ell\;[\mu\mathrm{K}^2]$",
                vertical_spacing=0.15 / nsurvey,
                horizontal_spacing=0.05 / nsurvey,
            )
            layout = self.base_layout.copy()
            layout.update(dict(height=1000))
            fig.update_layout(**layout)

            for i, name in enumerate(crosses.value):
                rowcol_kwargs = dict(row=indices[0][i] + 1, col=indices[1][i] + 1)
                inverse_rowcol_kwargs = dict(row=indices[1][i] + 1, col=indices[0][i] + 1)
                name1, name2 = name.split("x")
                fill_upper = mode[0] != mode[1] and name1 != name2 and nsurvey > 1

                # CMB + foregrounds
                if meta := self._get_db_entry(key=(name, "cmb_and_fg")):
                    cmb_and_fg_kwargs = dict(
                        name="cmb + foregrounds",
                        x=meta.ell,
                        y=meta.cmb_and_fg[mode],
                        line_color="black" if "white" in layout.get("template", "") else "white",
                        legendgroup="cmb_and_fg",
                        showlegend=i == 0,
                    )
                    fig.add_scatter(**cmb_and_fg_kwargs, **rowcol_kwargs)
                    if fill_upper:
                        fig.add_scatter(**cmb_and_fg_kwargs, **inverse_rowcol_kwargs)

                # CMB only
                if meta := self._get_db_entry(key="cmb"):
                    cmb_only_kwargs = dict(
                        name="cmb",
                        x=meta.ell,
                        y=meta.cmb[mode],
                        line_color="gray",
                        legendgroup="cmb",
                        showlegend=i == 0,
                    )
                    fig.add_scatter(**cmb_only_kwargs, **rowcol_kwargs)
                    if fill_upper:
                        fig.add_scatter(**cmb_only_kwargs, **inverse_rowcol_kwargs)

                # Individual foregrounds
                if meta := self._get_db_entry(key="foregrounds"):
                    all_fg_kwargs = dict(
                        name="all foregrounds",
                        x=meta.ell,
                        y=meta.fg_dict[mode.lower(), "all", name1, name2],
                        line_color="gray",
                        line_dash="dash",
                        legendgroup="all",
                        showlegend=i == 0,
                    )
                    fig.add_scatter(**all_fg_kwargs, **rowcol_kwargs)
                    if fill_upper:
                        all_fg_kwargs.update(
                            dict(y=meta.fg_dict[mode.lower(), "all", name2, name1])
                        )
                        fig.add_scatter(**all_fg_kwargs, **inverse_rowcol_kwargs)
                    for comp in fg_components[mode.lower()]:
                        individual_fg_kwargs = dict(
                            name=comp,
                            x=meta.ell,
                            y=meta.fg_dict[mode.lower(), comp, name1, name2],
                            line_color=fg_colors[comp],
                            legendgroup=comp,
                            showlegend=i == 0,
                        )
                        fig.add_scatter(**individual_fg_kwargs, **rowcol_kwargs)
                        if fill_upper:
                            individual_fg_kwargs.update(
                                dict(y=meta.fg_dict[mode.lower(), comp, name2, name1])
                            )
                            fig.add_scatter(**individual_fg_kwargs, **inverse_rowcol_kwargs)

                yaxes_kwargs = dict(
                    type="log" if mode == "TT" else "linear",
                    range=[-1, 4] if mode == "TT" else None,
                )
                fig.update_yaxes(**yaxes_kwargs, **rowcol_kwargs)
                if fill_upper:
                    fig.update_yaxes(**yaxes_kwargs, **inverse_rowcol_kwargs)

            return self._refresh_figure(change, fig)

        spectrum.observe(_update, names="value")
        crosses.observe(_update, names="value")

        if not fetch_data:
            self._update_tab(widgets.VBox([base_widget, _update()]))
        self.log.info(f"Directory '{directory}' loaded")

    @logger.capture()
    def _update_noise_model(self, fetch_data=False):
        if not (directory := self.directory_exists("noise_model")):
            self.log.debug("No noise model directory")
            return

        if self._add_tab(title="Noise model", callback=self._update_noise_model):
            return

        ell, nl_dict = best_fits.noise_dict_from_files(
            os.path.join(directory, "mean_{}x{}_{}_noise.dat"),
            sv_list=(surveys := d["surveys"]),
            arrays={sv: d[f"arrays_{sv}"] for sv in surveys},
            lmax=self.lmax,
            spectra=self.spectra,
            n_splits=None,
            # do not set to compare to original noise from data
            # {sv: d[f"n_splits_{sv}"] for sv in surveys},
        )
        for k, v in nl_dict.items():
            self._add_db_entry(
                key=("{0}_{1}x{0}_{2}".format(*k), "noise", "interpolate"), ell=ell, nl=v
            )

        # Get beams if not available
        if not self._has_db_entry("beam", flatten=True):
            self._update_beams(fetch_data=True)

        # Must be impossible but never know
        if not (spectra_dir := self.directory_exists("spectra")):
            self.log.debug("No spectra directory")
            return

        for name in self.cross_list:
            cl_type = d["type"]
            c1, c2 = name.split("x")
            lb, nbs_ar1xar1 = so_spectra.read_ps(
                os.path.join(spectra_dir, f"{cl_type}_{c1}x{c1}_noise.dat"), spectra=self.spectra
            )
            lb, nbs_ar1xar2 = so_spectra.read_ps(
                os.path.join(spectra_dir, f"{cl_type}_{c1}x{c2}_noise.dat"), spectra=self.spectra
            )
            b1 = self._get_db_entry(key=(c1, "beam"))
            b2 = self._get_db_entry(key=(c2, "beam"))

            bb_ar1, bb_ar2 = {}, {}
            for field in "TEB":
                lb, bb_ar1[field] = pspy_utils.naive_binning(
                    b1.ell, b1.bl[field], d["binning_file"], self.lmax
                )
                lb, bb_ar2[field] = pspy_utils.naive_binning(
                    b2.ell, b2.bl[field], d["binning_file"], self.lmax
                )

            for spec in self.spectra:
                X, Y = spec
                nbs_ar1xar1[spec] *= bb_ar1[X] * bb_ar1[Y]
                nbs_ar1xar2[spec] *= bb_ar1[X] * bb_ar2[Y]

            self._add_db_entry(
                key=(name, "noise", "data"), ell=lb, nl=nbs_ar1xar1 if c1 == c2 else nbs_ar1xar2
            )

        base_widget = widgets.HBox(
            [
                spectrum := widgets.Dropdown(
                    description="Mode",
                    options=self.spectra,
                    value=self.spectra[0],
                ),
                crosses := widgets.SelectMultiple(
                    description="Cross", options=self.cross_list, value=self.cross_list
                ),
            ]
        )
        layout = self.base_layout.copy()
        layout.update(dict(xaxis_title="$\ell$", yaxis_title="noise level [µK²]"))

        def _update(change=None):
            fig = go.Figure(layout=layout)
            for cross in crosses.value:
                if meta := self._get_db_entry((cross, "noise", "interpolate")):
                    fig.add_scatter(
                        name=cross,
                        x=meta.ell,
                        y=meta.nl[spectrum.value],
                        line_color=self.colors[cross],
                        legendgroup=cross,
                    )
                if meta := self._get_db_entry((cross, "noise", "data")):
                    fig.add_scatter(
                        name=cross,
                        x=meta.ell,
                        y=meta.nl[spectrum.value],
                        mode="markers",
                        marker_color=self.colors[cross],
                        legendgroup=cross,
                        showlegend=False,
                    )
            return self._refresh_figure(change, fig)

        crosses.observe(_update, names="value")
        spectrum.observe(_update, names="value")

        if not fetch_data:
            self._update_tab(widgets.VBox([base_widget, _update()]))
        self.log.info(f"Directory '{directory}' loaded")


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="psinspect", description="Power Spectrum Pipeline Inspector."
    )
    parser.add_argument(
        "dict_file", metavar="input_file.dict", default=None, help="A dict file to use.", nargs="?"
    )
    parser.add_argument(
        "--product-dir",
        default=None,
        help="The production directoryif different from the dict file.",
    )
    parser.add_argument(
        "-d", "--debug", help="Produce verbose debug output.", action="store_true", default=False
    )
    parser.add_argument("--version", action="version", version=psinspect.__version__)
    parser.add_argument(
        "--theme", help="Set default voila theme", default="light", choices=["light", "dark"]
    )
    arguments = parser.parse_args(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.DEBUG if arguments.debug else logging.INFO,
        force=True,
    )

    dict_file = arguments.dict_file
    if dict_file and not os.path.exists(dict_file):
        logging.error(f"The dict file '{dict_file}' does not exist!")
        raise SystemExit()

    product_dir = arguments.product_dir
    if product_dir and not os.path.exists(product_dir):
        logging.error(f"The production directory path '{product_dir}' does not exist!")
        raise SystemExit()

    # Here we set an env variable to be used by the notebooks. This sucks a bit but passing
    # parameter to notebook is not that easy : papermill exists and works fine but interacting with
    # voila seems broken
    # (https://voila.readthedocs.io/en/stable/customize.html#adding-the-hook-function-to-voila)
    if dict_file:
        os.environ[_psinspect_dict_file] = os.path.realpath(dict_file)
    if product_dir:
        os.environ[_psinspect_product_dir] = os.path.realpath(product_dir)
    os.environ[_psinspect_theme] = "plotly_white" if arguments.theme == "light" else "plotly_dark"
    os.environ[_psinspect_debug_flag] = str(arguments.debug)

    # create a voila instance
    app = Voila()

    notebook_path = os.path.join(os.path.dirname(psinspect.__file__), "app.ipynb")
    app.initialize([notebook_path])

    app.voila_configuration.show_tracebacks = arguments.debug
    app.voila_configuration.theme = arguments.theme

    app.start()
