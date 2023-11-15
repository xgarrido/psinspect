# import glob
import logging
import os
import pickle
import sys
from itertools import product
from numbers import Number

import ipywidgets as widgets
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
_psinspect_theme = "PSINSPECT_THEME"
_psinspect_debug_flag = "PSINSPECT_DEBUG_FLAG"

_product_dir = "."
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

banner = """
<center>
<svg xmlns="http://www.w3.org/2000/svg" height="10em" viewBox="0 0 512 512"><path d="M416 208c0 45.9-14.9 88.3-40 122.7L502.6 457.4c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L330.7 376c-34.4 25.2-76.8 40-122.7 40C93.1 416 0 322.9 0 208S93.1 0 208 0S416 93.1 416 208zM208 352a144 144 0 1 0 0-288 144 144 0 1 0 0 288z"/></svg>
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


class App:
    """An ipywidgets and plotly application for checking PSpipe productions"""

    def initialize(self, dict_file=None, debug=False):
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
        self._product_dir = self.file_chooser.selected_path
        self._dict_dump()

        self.log.info(f"Loading products from {_product_dir} directory...")
        self.tab = self.body = widgets.Tab(children=())

        if not hasattr(self, "app"):
            self._initialize_app()
        else:
            # Adding tab to app
            children = list(self.app.children)
            children[1] = self.body
            self.app.children = children

        self._update_info()
        self._update_passbands()
        self._update_beams()
        self._update_windows()
        self._update_maps()
        self._update_spectra()
        self._update_best_fits()
        self._update_noise_model()

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
        self.dict_dump = widgets.Output()
        self.dumper = widgets.RadioButtons(options=["print", "json", "yaml"])
        self.dumper.observe(self._dict_dump, names="value")
        self.footer = widgets.Accordion(
            children=[logger, self.dict_dump], titles=["Logs", "Dict dump"]
        )

    def _dict_dump(self, change=None):
        self.dict_dump.clear_output()
        with self.dict_dump:
            display(self.dumper)
            fmt = change.new if change else "print"
            if fmt == "print":
                import pprint

                pprint.pprint(d, sort_dicts=False)
            if fmt == "json":
                import json

                print(json.dumps(d, indent=2))
            if fmt == "yaml":
                import yaml

                print(yaml.dump(d))

    def _initialize_header(self):
        kwargs = {}
        if filename := os.getenv(_psinspect_dict_file, ""):
            kwargs = dict(
                path=os.path.dirname(filename),
                filename=os.path.basename(filename),
                select_default=True,
            )

        def _load_dict(chooser):
            d.read_from_file(chooser.selected)
            self.footer.selected_index = 0
            self.registered_callback.clear()
            self.updated_tab.clear()
            self.update()

        self.file_chooser = FileChooser(
            filter_pattern="*.dict", layout=widgets.Layout(height="auto", width="auto"), **kwargs
        )
        self.file_chooser.dir_icon = "/"
        self.file_chooser.register_callback(_load_dict)
        self.header = self.file_chooser
        if filename:
            _load_dict(self.file_chooser)

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
            key = tuple(key)
        self.db.setdefault(key, Bunch()).update(**value)

    def _get_db_entry(self, key):
        return self.db.get(tuple(key) if isinstance(key, (str, Number)) else key)

    def _has_db_entry(self, key):
        return key in sum(self.db.keys(), ())

    @logger.capture()
    def _update_info(self):
        self.spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
        self.lmax = d["lmax"]
        self.cross_list = pspipe_list.get_spec_name_list(d, delimiter="_")
        self.survey_list = pspipe_list.get_map_set_list(d)
        color_list = sns.color_palette(palette, n_colors=len(self.cross_list)).as_hex()
        self.colors = {name: color_list[i] for i, name in enumerate(self.cross_list)}
        self.colors.update({name: self.colors["{0}x{0}".format(name)] for name in self.survey_list})
        # self.db = {name: Bunch(color=color_list[i]) for i, name in enumerate(self.cross_list)}
        self.log.debug(f"survey list: {self.survey_list}")
        self.log.debug(f"cross list: {self.cross_list}")

    def directory_exists(self, dirname):
        return None if not os.path.exists(d := os.path.join(self._product_dir, dirname)) else d

    @logger.capture()
    def _update_passbands(self):
        _key = "passband"
        for survey in self.survey_list:
            if self._has_db_entry(key=(survey, _key)):
                continue
            nu_ghz, pb = None, None
            freq_info = d[f"freq_info_{survey}"]
            if d["do_bandpass_integration"]:
                if os.path.exists(filename := freq_info["passband"]):
                    nu_ghz, pb = np.loadtxt(filename).T
            else:
                nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.0])
            if nu_ghz is not None:
                self._add_db_entry(key=(survey, _key), nu_ghz=nu_ghz, pb=pb)

        if not self._has_db_entry(_key):
            self.log.debug("Passbands information unreachable")
            return

        if self._add_tab(title="Bandpass", callback=self._update_passbands):
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

    # Global method since beams can be use and needed by other part of the code (see
    # _update_noise_model for instance)
    def _has_beams(self):
        if self._has_db_entry(_key := "beam"):
            return True
        for survey in self.survey_list:
            if self._has_db_entry(key=(survey, _key)):
                continue
            if not os.path.exists(fn_beam_T := d[f"beam_T_{survey}"]) or not os.path.exists(
                fn_beam_pol := d[f"beam_pol_{survey}"]
            ):
                continue
            ell, bl = misc.read_beams(fn_beam_T, fn_beam_pol)
            idx = np.where((ell >= 2) & (ell < self.lmax))
            self._add_db_entry(key=(survey, _key), ell=ell, bl=bl, idx=idx)
        return self._has_db_entry(_key)

    @logger.capture()
    def _update_beams(self):
        if not self._has_beams():
            self.log.debug("Beams information unreachable")
            return

        if self._add_tab(title="Beams", callback=self._update_beams):
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
        layout = self.base_layout.copy()
        layout.update(dict(xaxis_title="$\ell$", yaxis_title="normalized beam"))
        fig = go.FigureWidget(layout=layout)

        def _update(change=None):
            fig.data = []
            for survey, mode in product(surveys.value, modes.value):
                if meta := self._get_db_entry((survey, "beam")):
                    fig.add_scatter(
                        name=f"{survey} - {mode}",
                        x=meta.ell[meta.idx],
                        y=meta.bl[mode][meta.idx],
                        line_color=self.colors[survey],
                        opacity=1 - mode_options.index(mode) / len(mode_options),
                    )

        surveys.observe(_update, names="value")
        modes.observe(_update, names="value")
        _update()

        self._update_tab(widgets.VBox([base_widget, fig]))

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
                if mode[0] != mode[1]:
                    subplot_titles[icol, irow] = "{1}x{0}".format(*name.split("x"))

            figure = make_subplots(
                rows=nsurvey,
                cols=nsurvey,
                shared_xaxes=True,
                shared_yaxes=True,
                subplot_titles=subplot_titles.flatten().tolist(),
                x_title="$\ell$",
                y_title="$D_\ell\;[\mu\mathrm{K}^2]$",
                vertical_spacing=0.15 / nsurvey,
                horizontal_spacing=0.05 / nsurvey,
            )
            figure.update_layout(**layout)

            for i, name in enumerate(crosses.value):
                name1, name2 = name.split("x")
                rowcol_kwargs = dict(row=indices[0][i] + 1, col=indices[1][i] + 1)
                inverse_rowcol_kwargs = dict(row=indices[1][i] + 1, col=indices[0][i] + 1)
                fill_upper = mode[0] != mode[1] and name1 != name2
                line_dash = {"cross": "solid", "auto": "dash", "noise": "dot"}
                for kind in kinds.value:
                    kind_kwargs = dict(
                        name=kind,
                        line_color=self.colors[name],
                        line_dash=line_dash[kind],
                        legendgroup=kind,
                    )

                    if meta := self._get_db_entry(key=(name, kind)):
                        figure.add_scatter(
                            x=meta.ell,
                            y=meta.spectra[mode],
                            showlegend=i == 0,
                            **kind_kwargs,
                            **rowcol_kwargs,
                        )
                        if fill_upper:
                            figure.add_scatter(
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
                        figure.add_scatter(
                            x=meta.ell, y=meta.spectra[mode], **split_kwargs, **rowcol_kwargs
                        )
                        if fill_upper:
                            figure.add_scatter(
                                x=meta.ell,
                                y=meta.spectra[mode[::-1]],
                                **split_kwargs,
                                **inverse_rowcol_kwargs,
                            )

                figure.update_yaxes(
                    type="log" if mode == "TT" else "linear",
                    range=[-1, 4] if mode == "TT" else None,
                    **rowcol_kwargs,
                )
            if change:
                tab = self.tab.children[self.tab.selected_index]
                children = list(tab.children)
                children[-1] = go.FigureWidget(figure)
                tab.children = children
            else:
                return go.FigureWidget(figure)

        spectrum.observe(_update, names="value")
        crosses.observe(_update, names="value")
        kinds.observe(_update, names="value")
        splits.observe(_update, names="value")

        self._update_tab(widgets.VBox([base_widget, _update()]))
        self.log.info(f"Directory '{directory}' loaded")

    @logger.capture()
    def _update_best_fits(self):
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
                if mode[0] != mode[1]:
                    subplot_titles[icol, irow] = "{1}x{0}".format(*name.split("x"))

            figure = make_subplots(
                rows=nsurvey,
                cols=nsurvey,
                shared_xaxes=True,
                shared_yaxes=True,
                subplot_titles=subplot_titles.flatten().tolist(),
                x_title="$\ell$",
                y_title="$D_\ell\;[\mu\mathrm{K}^2]$",
                vertical_spacing=0.15 / nsurvey,
                horizontal_spacing=0.05 / nsurvey,
            )
            figure.update_layout(**layout)

            for i, name in enumerate(crosses.value):
                name1, name2 = name.split("x")
                rowcol_kwargs = dict(row=indices[0][i] + 1, col=indices[1][i] + 1)
                inverse_rowcol_kwargs = dict(row=indices[1][i] + 1, col=indices[0][i] + 1)
                fill_upper = mode[0] != mode[1] and name1 != name2

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
                    figure.add_scatter(**cmb_and_fg_kwargs, **rowcol_kwargs)
                    if fill_upper:
                        figure.add_scatter(**cmb_and_fg_kwargs, **inverse_rowcol_kwargs)

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
                    figure.add_scatter(**cmb_only_kwargs, **rowcol_kwargs)
                    if fill_upper:
                        figure.add_scatter(**cmb_only_kwargs, **inverse_rowcol_kwargs)

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
                    figure.add_scatter(**all_fg_kwargs, **rowcol_kwargs)
                    if fill_upper:
                        all_fg_kwargs.update(
                            dict(y=meta.fg_dict[mode.lower(), "all", name2, name1])
                        )
                        figure.add_scatter(**all_fg_kwargs, **inverse_rowcol_kwargs)
                    for comp in fg_components[mode.lower()]:
                        individual_fg_kwargs = dict(
                            name=comp,
                            x=meta.ell,
                            y=meta.fg_dict[mode.lower(), comp, name1, name2],
                            line_color=fg_colors[comp],
                            legendgroup=comp,
                            showlegend=i == 0,
                        )
                        figure.add_scatter(**individual_fg_kwargs, **rowcol_kwargs)
                        if fill_upper:
                            individual_fg_kwargs.update(
                                dict(y=meta.fg_dict[mode.lower(), comp, name2, name1])
                            )
                            figure.add_scatter(**individual_fg_kwargs, **inverse_rowcol_kwargs)

                figure.update_yaxes(
                    type="log" if mode == "TT" else "linear",
                    range=[-1, 4] if mode == "TT" else None,
                    **rowcol_kwargs,
                )
            if change:
                tab = self.tab.children[self.tab.selected_index]
                children = list(tab.children)
                children[-1] = go.FigureWidget(figure)
                tab.children = children
            else:
                return go.FigureWidget(figure)

        spectrum.observe(_update, names="value")
        crosses.observe(_update, names="value")

        self._update_tab(widgets.VBox([base_widget, _update()]))
        self.log.info(f"Directory '{directory}' loaded")

    @logger.capture()
    def _update_noise_model(self):
        if not (directory := self.directory_exists("noise_model")):
            self.log.debug("No noise model directory")
            return

        if self._add_tab(title="Noise model", callback=self._update_noise_model):
            return

        if not self._has_db_entry(key=("noise", "interpolate")):
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

        if self._has_beams() and not self._has_db_entry(key=("noise", "data")):
            # Must be impossible but never know
            if not (spectra_dir := self.directory_exists("spectra")):
                self.log.debug("No spectra directory")
                return

            for name in self.cross_list:
                cl_type = d["type"]
                c1, c2 = name.split("x")
                lb, nbs_ar1xar1 = so_spectra.read_ps(
                    os.path.join(spectra_dir, f"{cl_type}_{c1}x{c1}_noise.dat"),
                    spectra=self.spectra,
                )
                lb, nbs_ar1xar2 = so_spectra.read_ps(
                    os.path.join(spectra_dir, f"{cl_type}_{c1}x{c2}_noise.dat"),
                    spectra=self.spectra,
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
                    key=(name, "noise", "data"),
                    ell=lb,
                    nl=nbs_ar1xar1 if c1 == c2 else nbs_ar1xar2,
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
        fig = go.FigureWidget(layout=layout)

        def _update(change=None):
            fig.data = []
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

        crosses.observe(_update, names="value")
        spectrum.observe(_update, names="value")
        _update()

        self._update_tab(widgets.VBox([base_widget, fig]))
        self.log.info(f"Directory '{directory}' loaded")


def run(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="psinspect", description="Power Spectrum Pipeline Inspector."
    )
    parser.add_argument(
        "dict_file", metavar="input_file.dict", default="", help="A dict file to use.", nargs="?"
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

    # Here we set an env variable to be used by the notebooks. This sucks a bit but passing
    # parameter to notebook is not that easy : papermill exists and works fine but interacting with
    # voila seems broken
    # (https://voila.readthedocs.io/en/stable/customize.html#adding-the-hook-function-to-voila)
    if dict_file:
        os.environ[_psinspect_dict_file] = os.path.realpath(dict_file)
    os.environ[_psinspect_theme] = "plotly_white" if arguments.theme == "light" else "plotly_dark"
    os.environ[_psinspect_debug_flag] = str(arguments.debug)

    # create a voila instance
    app = Voila()

    notebook_path = os.path.join(os.path.dirname(psinspect.__file__), "app.ipynb")
    app.initialize([notebook_path])

    app.voila_configuration.show_tracebacks = arguments.debug
    app.voila_configuration.theme = arguments.theme

    app.start()
