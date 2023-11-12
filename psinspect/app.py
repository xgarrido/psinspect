# import glob
import logging
import os
import sys
from itertools import product

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from ipyfilechooser import FileChooser
from IPython.display import HTML, display
from pspipe import conventions as cvt
from pspipe_utils import best_fits, log, misc, pspipe_list
from pspy import so_dict
from voila.app import Voila

import psinspect
from psinspect._version import version

_psinspect_dict_file = "PSINSPECT_DICT_FILE"
_psinspect_debug_flag = "PSINSPECT_DEBUG_FLAG"

d = so_dict.so_dict()

base_layout = dict(
    height=800,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


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


def directory_exists(dirname):
    return None if not os.path.exists(d := os.path.join(cvt._product_dir, dirname)) else d


# Global log widget to catch message
logger = widgets.Output()

banner = """
<center>
<svg xmlns="http://www.w3.org/2000/svg" height="10em" viewBox="0 0 512 512"><path d="M416 208c0 45.9-14.9 88.3-40 122.7L502.6 457.4c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L330.7 376c-34.4 25.2-76.8 40-122.7 40C93.1 416 0 322.9 0 208S93.1 0 208 0S416 93.1 416 208zM208 352a144 144 0 1 0 0-288 144 144 0 1 0 0 288z"/></svg>
<p>
<b><font size="+3">PS Pipeline Inspector</font></b>
</p>
<p>
<font size="+1">Select a dict file</font>
</p>
</center>
"""


class App:
    """An ipywidgets and plotly application for checking PSpipe productions"""

    def initialize(self, dict_file=None, debug=False):
        # This is to fix loading of mathjax that is not correctly done and make the application totally
        # bugging see https://github.com/microsoft/vscode-jupyter/issues/8131#issuecomment-1589961116
        display(
            HTML(
                '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
            )
        )
        self.log = log.get_logger(
            debug=bool(os.getenv(_psinspect_debug_flag, debug)),
            fmt="%(asctime)s - %(levelname)s: %(message)s",
        )
        # To avoid red background
        # https://stackoverflow.com/questions/25360714/avoid-red-background-color-for-logging-output-in-ipython-notebook
        self.log.handlers[0].stream = sys.stdout

        self.registered_callback = []
        self.updated_tab = set()
        if dict_file:
            os.environ[_psinspect_dict_file] = os.path.realpath(dict_file)

        # Initialize first the footer i.e. the dumper and then the loader since a dict file can be
        # passed as argument and then will need the dumper to work
        self._initialize_footer()
        self._initialize_header()
        self._initialize_app()

    def run(self):
        display(self.app)

    @logger.capture()
    def update(self):
        cvt._product_dir = self.file_chooser.selected_path
        self._dict_dump()

        self.log.info(f"Loading products from {cvt._product_dir} directory...")
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
        self._update_best_fits()

        def _update(change=None):
            if (index := self.tab.selected_index) in self.updated_tab:
                return
            self.registered_callback[index]()
            self.updated_tab.add(index)

        self.tab.observe(_update, names="selected_index")
        if self.tab.children:
            _update()
        self.log.info("Updating done")

    def _initialize_footer(self):
        self.dict_dump = widgets.Output()
        self.dumper = widgets.RadioButtons(options=["print", "json", "yaml"])
        self.dumper.observe(self._dict_dump, names="value")
        self.footer = widgets.Accordion(
            children=[logger, self.dict_dump],
            titles=["Logs", "Dict dump"],
            selected_index=0,
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
            self.body = widgets.HTML(value=banner)
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

    @logger.capture()
    def _update_info(self):
        self.spectra = cvt.spectra
        self.lmax = d["lmax"]
        self.cross_list = pspipe_list.get_spec_name_list(d, delimiter="_")
        self.survey_list = pspipe_list.get_map_set_list(d)
        color_list = sns.color_palette("deep", n_colors=len(self.cross_list)).as_hex()
        self.db = {name: Bunch(color=color_list[i]) for i, name in enumerate(self.cross_list)}
        self.log.debug(f"survey list: {self.survey_list}")
        self.log.debug(f"cross list: {self.cross_list}")

    @logger.capture()
    def _update_passbands(self):
        passbands = {}
        for survey in self.survey_list:
            nu_ghz, pb = None, None
            freq_info = d[f"freq_info_{survey}"]
            if d["do_bandpass_integration"]:
                if os.path.exists(filename := freq_info["passband"]):
                    nu_ghz, pb = np.loadtxt(filename).T
            else:
                nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.0])
            if nu_ghz is not None:
                passbands[survey] = Bunch(
                    nu_ghz=nu_ghz, pb=pb, color=self.db["{0}x{0}".format(survey)].color
                )

        if not passbands:
            self.log.debug("Passbands information unreachable")
            return

        if self._add_tab(title="Bandpass", callback=self._update_passbands):
            return

        layout = base_layout.copy()
        layout.update(dict(xaxis_title="frequency [GHz]", yaxis_title="transmission"))
        fig = go.FigureWidget(layout=layout)
        for survey, meta in passbands.items():
            fig.add_scatter(
                name=survey,
                x=meta.nu_ghz,
                y=meta.pb,
                mode="lines",
                line=dict(color=meta.color),
            )

        self._update_tab(fig)

    @logger.capture()
    def _update_beams(self):
        beams = {}
        for survey in self.survey_list:
            if not os.path.exists(fn_beam_T := d[f"beam_T_{survey}"]):
                continue
            if not os.path.exists(fn_beam_pol := d[f"beam_pol_{survey}"]):
                continue
            ell, bl = misc.read_beams(fn_beam_T, fn_beam_pol)
            idx = np.where((ell >= 2) & (ell < self.lmax))
            beams[survey] = Bunch(
                ell=ell, bl=bl, idx=idx, color=self.db["{0}x{0}".format(survey)].color
            )

        if not beams:
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
        layout = base_layout.copy()
        layout.update(dict(xaxis_title="$\ell$", yaxis_title="normalized beam"))
        fig = go.FigureWidget(layout=layout)

        def _update(change=None):
            fig.data = []
            for survey, mode in product(surveys.value, modes.value):
                self.log.debug(f"{survey} - {mode}")
                beam = beams[survey]
                fig.add_scatter(
                    name=f"{survey} - {mode}",
                    x=beam.ell[beam.idx],
                    y=beam.bl[mode][idx],
                    mode="lines",
                    line=dict(color=beam.color),
                    opacity=1 - mode_options.index(mode) / len(mode_options),
                )

        surveys.observe(_update, names="value")
        modes.observe(_update, names="value")
        _update()

        self._update_tab(widgets.VBox([base_widget, fig]))

    @logger.capture()
    def _update_windows(self):
        self.log.debug("Entering _update_windows")
        if not (directory := directory_exists("windows")):
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
        if not (directory := directory_exists("plots/maps")):
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
    def _update_best_fits(self):
        self.log.debug("Entering _update_best_fits")
        if not (directory := directory_exists("best_fits")):
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
            self.db[name].update(ell=ell, cmb_and_fg=cmb_and_fg_dict)

        layout = base_layout.copy()
        layout.update(dict(xaxis_title="$\ell$", yaxis_title=r"$D_\ell\;[\mu\mathrm{K}^2]$"))
        fig = go.FigureWidget(layout=layout)

        def _update(change=None):
            fig.data = []
            fig.update_yaxes(type="log" if spectrum.value == "TT" else "linear")
            for name, meta in self.db.items():
                fig.add_scatter(
                    name=name,
                    x=meta.ell,
                    y=meta.cmb_and_fg[spectrum.value],
                    mode="lines",
                    line=dict(color=meta.color),
                )

        spectrum = widgets.Dropdown(
            value=self.spectra[0], options=self.spectra, description="Spectrum"
        )
        spectrum.observe(_update, names="value")
        _update()

        self._update_tab(widgets.VBox([spectrum, fig]))
        self.log.info(f"Directory '{directory}' loaded")


#         ###


#         fg_dict = best_fits.get_foreground_dict(
#             l_th, passbands, d["fg_components"], d["fg_params"], d["fg_norm"]
#         )

#         ell, fg_dict = best_fits.fg_dict_from_files(
#             os.path.join(dir, "fg_{}x{}.dat"), survey_list, lmax=lmax, spectra=spectra
#         )
#         print(fg_dict)
#         for name in cross_list:
#             db[name].update(ell_fg=ell, fg=fg_dict[*name.split("x")])

#         fg_components = d["fg_components"]
#         for spec in spectra:
#             if (name := "tSZ_and_CIB") in (fg := fg_components.get(spec.lower(), [])):
#                 fg.remove(name)
#                 fg += ["tSZ", "cibc", "tSZxCIB"]

#         def _update(change=None):
#             fig.data = []
#             fig.update_yaxes(type="log" if spectrum.value == "TT" else "linear")

#             # for comp in fg_components[spectrum.value.lower()]:
#             #     fig.add_scatter(
#             #         name=comp,
#             #         x=db[surveys.value].ell_fg,
#             #         y=db[surveys.value].fg[comp],
#             #         mode="lines",
#             #         line=dict(color=meta.color),
#             #     )
#             logging.info(fg_components)

#             return

#         spectrum = widgets.Dropdown(
#             options=(options := ["TT", "TE", "TB", "EE", "EB", "BB"]),
#             value=options[0],
#             description="Spectrum",
#         )
#         surveys = widgets.Dropdown(value=survey_list[0], options=survey_list, description="Survey")
#         spectrum.observe(_update, names="value")
#         surveys.observe(_update, names="value")
#         _update()

#         add_tab(widgets.VBox([widgets.HBox([surveys, spectrum]), fig]), "Foregrounds")
#         logging.info(f"Directory {dir} loaded")


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
    parser.add_argument("--version", action="version", version=version)
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
    os.environ[_psinspect_debug_flag] = str(arguments.debug)

    # create a voila instance
    app = Voila()

    notebook_path = os.path.join(os.path.dirname(psinspect.__file__), "app.ipynb")
    app.initialize([notebook_path])

    app.voila_configuration.show_tracebacks = arguments.debug

    app.start()
