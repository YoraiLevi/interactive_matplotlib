from collections import OrderedDict
from collections.abc import MutableSequence
from contextlib import ExitStack
import functools
import inspect
import itertools
import logging
from numbers import Real
from operator import attrgetter
import types

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, docstring
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

def _process_plot_format(fmt):
    """
    Convert a MATLAB style color/line style format string to a (*linestyle*,
    *marker*, *color*) tuple.

    Example format strings include:

    * 'ko': black circles
    * '.b': blue dots
    * 'r--': red dashed lines
    * 'C2--': the third color in the color cycle, dashed lines

    The format is absolute in the sense that if a linestyle or marker is not
    defined in *fmt*, there is no line or marker. This is expressed by
    returning 'None' for the respective quantity.

    See Also
    --------
    matplotlib.Line2D.lineStyles, matplotlib.colors.cnames
        All possible styles and color format strings.
    """

    linestyle = None
    marker = None
    color = None

    # Is fmt just a colorspec?
    try:
        color = mcolors.to_rgba(fmt)

        # We need to differentiate grayscale '1.0' from tri_down marker '1'
        try:
            fmtint = str(int(fmt))
        except ValueError:
            return linestyle, marker, color  # Yes
        else:
            if fmt != fmtint:
                # user definitely doesn't want tri_down marker
                return linestyle, marker, color  # Yes
            else:
                # ignore converted color
                color = None
    except ValueError:
        pass  # No, not just a color.

    i = 0
    while i < len(fmt):
        c = fmt[i]
        if fmt[i:i+2] in mlines.lineStyles:  # First, the two-char styles.
            if linestyle is not None:
                raise ValueError(
                    'Illegal format string "%s"; two linestyle symbols' % fmt)
            linestyle = fmt[i:i+2]
            i += 2
        elif c in mlines.lineStyles:
            if linestyle is not None:
                raise ValueError(
                    'Illegal format string "%s"; two linestyle symbols' % fmt)
            linestyle = c
            i += 1
        elif c in mlines.lineMarkers:
            if marker is not None:
                raise ValueError(
                    'Illegal format string "%s"; two marker symbols' % fmt)
            marker = c
            i += 1
        elif c in mcolors.get_named_colors_mapping():
            if color is not None:
                raise ValueError(
                    'Illegal format string "%s"; two color symbols' % fmt)
            color = c
            i += 1
        elif c == 'C' and i < len(fmt) - 1:
            color_cycle_number = int(fmt[i + 1])
            color = mcolors.to_rgba("C{}".format(color_cycle_number))
            i += 2
        else:
            raise ValueError(
                'Unrecognized character %c in format string' % c)

    if linestyle is None and marker is None:
        linestyle = mpl.rcParams['lines.linestyle']
    if linestyle is None:
        linestyle = 'None'
    if marker is None:
        marker = 'None'

    return linestyle, marker, color


class _process_plot_var_args:
    """
    Process variable length arguments to `~.Axes.plot`, to support ::

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of *x*, *y*, *fmt* are allowed
    """
    def __init__(self, axes, command='plot'):
        self.axes = axes
        self.command = command
        self.set_prop_cycle(None)

    def __getstate__(self):
        # note: it is not possible to pickle a generator (and thus a cycler).
        return {'axes': self.axes, 'command': self.command}

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.set_prop_cycle(None)

    def set_prop_cycle(self, cycler):
        if cycler is None:
            cycler = mpl.rcParams['axes.prop_cycle']
        self.prop_cycler = itertools.cycle(cycler)
        self._prop_keys = cycler.keys  # This should make a copy

    def __call__(self, *args, data=None, **kwargs):
        self.axes._process_unit_info(kwargs=kwargs)

        for pos_only in "xy":
            if pos_only in kwargs:
                raise TypeError("{} got an unexpected keyword argument {!r}"
                                .format(self.command, pos_only))

        if not args:
            return

        if data is None:  # Process dict views
            args = [cbook.sanitize_sequence(a) for a in args]
        else:  # Process the 'data' kwarg.
            replaced = [mpl._replacer(data, arg) for arg in args]
            if len(args) == 1:
                label_namer_idx = 0
            elif len(args) == 2:  # Can be x, y or y, c.
                # Figure out what the second argument is.
                # 1) If the second argument cannot be a format shorthand, the
                #    second argument is the label_namer.
                # 2) Otherwise (it could have been a format shorthand),
                #    a) if we did perform a substitution, emit a warning, and
                #       use it as label_namer.
                #    b) otherwise, it is indeed a format shorthand; use the
                #       first argument as label_namer.
                try:
                    _process_plot_format(args[1])
                except ValueError:  # case 1)
                    label_namer_idx = 1
                else:
                    if replaced[1] is not args[1]:  # case 2a)
                        _api.warn_external(
                            f"Second argument {args[1]!r} is ambiguous: could "
                            f"be a format string but is in 'data'; using as "
                            f"data.  If it was intended as data, set the "
                            f"format string to an empty string to suppress "
                            f"this warning.  If it was intended as a format "
                            f"string, explicitly pass the x-values as well.  "
                            f"Alternatively, rename the entry in 'data'.",
                            RuntimeWarning)
                        label_namer_idx = 1
                    else:  # case 2b)
                        label_namer_idx = 0
            elif len(args) == 3:
                label_namer_idx = 1
            else:
                raise ValueError(
                    "Using arbitrary long args with data is not supported due "
                    "to ambiguity of arguments; use multiple plotting calls "
                    "instead")
            if kwargs.get("label") is None:
                kwargs["label"] = mpl._label_from_arg(
                    replaced[label_namer_idx], args[label_namer_idx])
            args = replaced

        if len(args) >= 4 and not cbook.is_scalar_or_string(
                kwargs.get("label")):
            raise ValueError("plot() with multiple groups of data (i.e., "
                             "pairs of x and y) does not support multiple "
                             "labels")

        # Repeatedly grab (x, y) or (x, y, format) from the front of args and
        # massage them into arguments to plot() or fill().

        while args:
            this, args = args[:2], args[2:]
            if args and isinstance(args[0], str):
                this += args[0],
                args = args[1:]
            yield from self._plot_args(this, kwargs)

    def get_next_color(self):
        """Return the next color in the cycle."""
        if 'color' not in self._prop_keys:
            return 'k'
        return next(self.prop_cycler)['color']

    def _getdefaults(self, ignore, kw):
        """
        If some keys in the property cycle (excluding those in the set
        *ignore*) are absent or set to None in the dict *kw*, return a copy
        of the next entry in the property cycle, excluding keys in *ignore*.
        Otherwise, don't advance the property cycle, and return an empty dict.
        """
        prop_keys = self._prop_keys - ignore
        if any(kw.get(k, None) is None for k in prop_keys):
            # Need to copy this dictionary or else the next time around
            # in the cycle, the dictionary could be missing entries.
            default_dict = next(self.prop_cycler).copy()
            for p in ignore:
                default_dict.pop(p, None)
        else:
            default_dict = {}
        return default_dict

    def _setdefaults(self, defaults, kw):
        """
        Add to the dict *kw* the entries in the dict *default* that are absent
        or set to None in *kw*.
        """
        for k in defaults:
            if kw.get(k, None) is None:
                kw[k] = defaults[k]

    def _makeline(self, x, y, kw, kwargs):
        kw = {**kw, **kwargs}  # Don't modify the original kw.
        default_dict = self._getdefaults(set(), kw)
        self._setdefaults(default_dict, kw)
        seg = mlines.Line2D(x, y, **kw)
        return seg, kw

    def _makefill(self, x, y, kw, kwargs):
        # Polygon doesn't directly support unitized inputs.
        x = self.axes.convert_xunits(x)
        y = self.axes.convert_yunits(y)

        kw = kw.copy()  # Don't modify the original kw.
        kwargs = kwargs.copy()

        # Ignore 'marker'-related properties as they aren't Polygon
        # properties, but they are Line2D properties, and so they are
        # likely to appear in the default cycler construction.
        # This is done here to the defaults dictionary as opposed to the
        # other two dictionaries because we do want to capture when a
        # *user* explicitly specifies a marker which should be an error.
        # We also want to prevent advancing the cycler if there are no
        # defaults needed after ignoring the given properties.
        ignores = {'marker', 'markersize', 'markeredgecolor',
                   'markerfacecolor', 'markeredgewidth'}
        # Also ignore anything provided by *kwargs*.
        for k, v in kwargs.items():
            if v is not None:
                ignores.add(k)

        # Only using the first dictionary to use as basis
        # for getting defaults for back-compat reasons.
        # Doing it with both seems to mess things up in
        # various places (probably due to logic bugs elsewhere).
        default_dict = self._getdefaults(ignores, kw)
        self._setdefaults(default_dict, kw)

        # Looks like we don't want "color" to be interpreted to
        # mean both facecolor and edgecolor for some reason.
        # So the "kw" dictionary is thrown out, and only its
        # 'color' value is kept and translated as a 'facecolor'.
        # This design should probably be revisited as it increases
        # complexity.
        facecolor = kw.get('color', None)

        # Throw out 'color' as it is now handled as a facecolor
        default_dict.pop('color', None)

        # To get other properties set from the cycler
        # modify the kwargs dictionary.
        self._setdefaults(default_dict, kwargs)

        seg = mpatches.Polygon(np.column_stack((x, y)),
                               facecolor=facecolor,
                               fill=kwargs.get('fill', True),
                               closed=kw['closed'])
        seg.set(**kwargs)
        return seg, kwargs

    def _plot_args(self, tup, kwargs, return_kwargs=False):
        """
        Process the arguments of ``plot([x], y, [fmt], **kwargs)`` calls.

        This processes a single set of ([x], y, [fmt]) parameters; i.e. for
        ``plot(x, y, x2, y2)`` it will be called twice. Once for (x, y) and
        once for (x2, y2).

        x and y may be 2D and thus can still represent multiple datasets.

        For multiple datasets, if the keyword argument *label* is a list, this
        will unpack the list and assign the individual labels to the datasets.

        Parameters
        ----------
        tup : tuple
            A tuple of the positional parameters. This can be one of

            - (y,)
            - (x, y)
            - (y, fmt)
            - (x, y, fmt)

        kwargs : dict
            The keyword arguments passed to ``plot()``.

        return_kwargs : bool
            If true, return the effective keyword arguments after label
            unpacking as well.

        Returns
        -------
        result
            If *return_kwargs* is false, a list of Artists representing the
            dataset(s).
            If *return_kwargs* is true, a list of (Artist, effective_kwargs)
            representing the dataset(s). See *return_kwargs*.
            The Artist is either `.Line2D` (if called from ``plot()``) or
            `.Polygon` otherwise.
        """
        if len(tup) > 1 and isinstance(tup[-1], str):
            # xy is tup with fmt stripped (could still be (y,) only)
            *xy, fmt = tup
            linestyle, marker, color = _process_plot_format(fmt)
        elif len(tup) == 3:
            raise ValueError('third arg must be a format string')
        else:
            xy = tup
            linestyle, marker, color = None, None, None

        # Don't allow any None value; these would be up-converted to one
        # element array of None which causes problems downstream.
        if any(v is None for v in tup):
            raise ValueError("x, y, and format string must not be None")

        kw = {}
        for prop_name, val in zip(('linestyle', 'marker', 'color'),
                                  (linestyle, marker, color)):
            if val is not None:
                # check for conflicts between fmt and kwargs
                if (fmt.lower() != 'none'
                        and prop_name in kwargs
                        and val != 'None'):
                    # Technically ``plot(x, y, 'o', ls='--')`` is a conflict
                    # because 'o' implicitly unsets the linestyle
                    # (linestyle='None').
                    # We'll gracefully not warn in this case because an
                    # explicit set via kwargs can be seen as intention to
                    # override an implicit unset.
                    # Note: We don't val.lower() != 'none' because val is not
                    # necessarily a string (can be a tuple for colors). This
                    # is safe, because *val* comes from _process_plot_format()
                    # which only returns 'None'.
                    _api.warn_external(
                        f"{prop_name} is redundantly defined by the "
                        f"'{prop_name}' keyword argument and the fmt string "
                        f'"{fmt}" (-> {prop_name}={val!r}). The keyword '
                        f"argument will take precedence.")
                kw[prop_name] = val

        if len(xy) == 2:
            x = _check_1d(xy[0])
            y = _check_1d(xy[1])
        else:
            x, y = index_of(xy[-1])

        if self.axes.xaxis is not None:
            self.axes.xaxis.update_units(x)
        if self.axes.yaxis is not None:
            self.axes.yaxis.update_units(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have same first dimension, but "
                             f"have shapes {x.shape} and {y.shape}")
        if x.ndim > 2 or y.ndim > 2:
            raise ValueError(f"x and y can be no greater than 2D, but have "
                             f"shapes {x.shape} and {y.shape}")
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.command == 'plot':
            make_artist = self._makeline
        else:
            kw['closed'] = kwargs.get('closed', True)
            make_artist = self._makefill

        ncx, ncy = x.shape[1], y.shape[1]
        if ncx > 1 and ncy > 1 and ncx != ncy:
            raise ValueError(f"x has {ncx} columns but y has {ncy} columns")
        if ncx == 0 or ncy == 0:
            return []

        label = kwargs.get('label')
        n_datasets = max(ncx, ncy)
        if n_datasets > 1 and not cbook.is_scalar_or_string(label):
            if len(label) != n_datasets:
                raise ValueError(f"label must be scalar or have the same "
                                 f"length as the input data, but found "
                                 f"{len(label)} for {n_datasets} datasets.")
            labels = label
        else:
            labels = [label] * n_datasets

        result = (make_artist(x[:, j % ncx], y[:, j % ncy], kw,
                              {**kwargs, 'label': label})
                  for j, label in enumerate(labels))

        if return_kwargs:
            return list(result)
        else:
            return [l[0] for l in result]


def cla(self):
        """Clear the Axes."""
        # Note: this is called by Axes.__init__()

        # stash the current visibility state
        if hasattr(self, 'patch'):
            patch_visible = self.patch.get_visible()
        else:
            patch_visible = True

        xaxis_visible = self.xaxis.get_visible()
        yaxis_visible = self.yaxis.get_visible()

        # self.xaxis.clear()
        # self.yaxis.clear()

        for name, spine in self.spines.items():
            spine.clear()

        self.ignore_existing_data_limits = True
        # self.callbacks = cbook.CallbackRegistry()

        if self._sharex is not None:
            self.sharex(self._sharex)
        else:
            self.xaxis._set_scale('linear')
            try:
                pass
                # self.set_xlim(0, 1)
            except TypeError:
                pass
        if self._sharey is not None:
            self.sharey(self._sharey)
        else:
            self.yaxis._set_scale('linear')
            try:
                pass
                # self.set_ylim(0, 1)
            except TypeError:
                pass

        # update the minor locator for x and y axis based on rcParams
        if mpl.rcParams['xtick.minor.visible']:
            self.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        if mpl.rcParams['ytick.minor.visible']:
            self.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        if self._sharex is None:
            self._autoscaleXon = True
        if self._sharey is None:
            self._autoscaleYon = True
        self._xmargin = mpl.rcParams['axes.xmargin']
        self._ymargin = mpl.rcParams['axes.ymargin']
        self._tight = None
        self._use_sticky_edges = True
        self._update_transScale()  # needed?

        self._get_lines = _process_plot_var_args(self)
        self._get_patches_for_fill = _process_plot_var_args(self, 'fill')

        self._gridOn = mpl.rcParams['axes.grid']
        self._children = []
        self._mouseover_set = _OrderedSet()
        self.child_axes = []
        self._current_image = None  # strictly for pyplot via _sci, _gci
        self._projection_init = None  # strictly for pyplot.subplot
        self.legend_ = None
        self.containers = []

        self.grid(False)  # Disable grid on init to use rcParameter
        self.grid(self._gridOn, which=mpl.rcParams['axes.grid.which'],
                  axis=mpl.rcParams['axes.grid.axis'])
        props = font_manager.FontProperties(
            size=mpl.rcParams['axes.titlesize'],
            weight=mpl.rcParams['axes.titleweight'])

        y = mpl.rcParams['axes.titley']
        if y is None:
            y = 1.0
            self._autotitlepos = True
        else:
            self._autotitlepos = False

        self.title = mtext.Text(
            x=0.5, y=y, text='',
            fontproperties=props,
            verticalalignment='baseline',
            horizontalalignment='center',
            )
        self._left_title = mtext.Text(
            x=0.0, y=y, text='',
            fontproperties=props.copy(),
            verticalalignment='baseline',
            horizontalalignment='left', )
        self._right_title = mtext.Text(
            x=1.0, y=y, text='',
            fontproperties=props.copy(),
            verticalalignment='baseline',
            horizontalalignment='right',
            )
        title_offset_points = mpl.rcParams['axes.titlepad']
        # refactor this out so it can be called in ax.set_title if
        # pad argument used...
        self._set_title_offset_trans(title_offset_points)

        for _title in (self.title, self._left_title, self._right_title):
            self._set_artist_props(_title)

        # The patch draws the background of the Axes.  We want this to be below
        # the other artists.  We use the frame to draw the edges so we are
        # setting the edgecolor to None.
        self.patch = self._gen_axes_patch()
        self.patch.set_figure(self.figure)
        self.patch.set_facecolor(self._facecolor)
        self.patch.set_edgecolor('none')
        self.patch.set_linewidth(0)
        self.patch.set_transform(self.transAxes)

        self.set_axis_on()

        self.xaxis.set_clip_path(self.patch)
        self.yaxis.set_clip_path(self.patch)

        self._shared_axes["x"].clean()
        self._shared_axes["y"].clean()
        if self._sharex is not None:
            self.xaxis.set_visible(xaxis_visible)
            self.patch.set_visible(patch_visible)
        if self._sharey is not None:
            self.yaxis.set_visible(yaxis_visible)
            self.patch.set_visible(patch_visible)

        self.stale = True