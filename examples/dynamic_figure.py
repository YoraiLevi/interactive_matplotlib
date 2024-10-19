# https://github.com/mpl-extensions/mpl-pan-zoom/blob/823829ea774655701ffa6bbded1a97b4a9bf6f23/mpl_pan_zoom/_zoom.py#L9
def zoom_factory(fig, base_scale=1.1, listener = None):
    """
    Add ability to zoom with the scroll wheel.


    Parameters
    ----------
    ax : matplotlib axes object
        axis on which to implement scroll to zoom
    base_scale : float
        how much zoom on each tick of scroll wheel

    Returns
    -------
    disconnect_zoom : function
        call this to disconnect the scroll listener
    """
    if hasattr(fig.canvas, "capture_scroll"):
        fig.canvas.capture_scroll = True
    has_toolbar = hasattr(fig.canvas, "toolbar") and fig.canvas.toolbar is not None
    if has_toolbar:
        # it might be possible to have an interactive backend without
        # a toolbar. I'm not sure so being safe here
        toolbar = fig.canvas.toolbar

    def zoom_fun(event):
        if has_toolbar:
            toolbar.push_current()
        ax = event.inaxes
        if event.inaxes is None:
            return
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == "up":
            # deal with zoom in
            scale_factor = base_scale
        elif event.button == "down":
            # deal with zoom out
            scale_factor = 1 / base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        new_xlim = [
            xdata - (xdata - cur_xlim[0]) / scale_factor,
            xdata + (cur_xlim[1] - xdata) / scale_factor,
        ]
        new_ylim = [
            ydata - (ydata - cur_ylim[0]) / scale_factor,
            ydata + (cur_ylim[1] - ydata) / scale_factor,
        ]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        if listener is not None:
            listener(event)
        # ax.figure.canvas.draw_idle()  # force re-draw

    # attach the call back
    cid = fig.canvas.mpl_connect("scroll_event", zoom_fun)

    def disconnect_zoom():
        fig.canvas.mpl_disconnect(cid)

    # return the disconnect function
    return disconnect_zoom

def pan_factory(fig,button=3,listener=None):
    """
    Enable panning a plot with any mouse button.
    Parameters
    ----------
    button : int
        Determines which button will be used (default right click).
        Left: 1
        Middle: 2
        Right: 3
    """
    _id_drag = None
    _id_press = None
    _id_release = None
    _xypress = []
    self = object()

    def enabled() -> bool:
        """
        Status of the PanManager, whether it's enabled or disabled.
        """
        return _id_press is not None and _id_release is not None

    def enable():
        nonlocal _id_press, _id_release, _id_drag, _xypress
        """
        Enable the PanManager. It should not be necessary to call this function
        unless it's used after a call to :meth:`PanManager.disable`.

        Raises
        ------
        RuntimeError
            If the PanManager is already enabled.
        """
        if enabled():
            raise RuntimeError("The PanManager is already enabled")
    
        _id_press = fig.canvas.mpl_connect("button_press_event", press)
        _id_release = fig.canvas.mpl_connect("button_release_event", release)

    def disable():
        nonlocal _id_press, _id_release, _id_drag, _xypress
        """
        Disable the PanManager.

        Raises
        ------
        RuntimeError
            If the PanManager is already disabled.
        """
        nonlocal _id_press, _id_release, _id_drag
        
        if not enabled:
            raise RuntimeError("The PanManager is already disabled")

        fig.canvas.mpl_disconnect(_id_press)
        fig.canvas.mpl_disconnect(_id_release)

        _id_press = None
        _id_release = None
        # just to be sure
        if fig.canvas.widgetlock.isowner(self):
            fig.canvas.widgetlock.release(self)

    def _cancel_action():
        nonlocal _id_press, _id_release, _id_drag, _xypress
        
        _xypress = []
        if _id_drag:
            fig.canvas.mpl_disconnect(_id_drag)
            _id_drag = None
        if fig.canvas.widgetlock.isowner(self):
            fig.canvas.widgetlock.release(self)

    def press(event):
        nonlocal _id_press, _id_release, _id_drag, _xypress
        # print(f"press event.button: {event.button}")
        if event.button != button:
            _cancel_action()
            return
        if not fig.canvas.widgetlock.available(self):
            return

        fig.canvas.widgetlock(self)

        x, y = event.x, event.y

        _xypress = []
        for i, a in enumerate(fig.get_axes()):
            if (
                x is not None
                and y is not None
                and a.in_axes(event)
                and a.get_navigate()
                and a.can_pan()
            ):
                a.start_pan(x, y, event.button)
                _xypress.append((a, i))
                _id_drag = fig.canvas.mpl_connect("motion_notify_event", _mouse_move)

    def release(event):
        nonlocal _id_press, _id_release, _id_drag, _xypress
        # print(f"release event.button: {event.button}")
        _cancel_action()
        fig.canvas.mpl_disconnect(_id_drag)

        for a, _ind in _xypress:
            a.end_pan()
        if not _xypress:
            _cancel_action()
            return
        _cancel_action()

    def _mouse_move(event):
        nonlocal _id_press, _id_release, _id_drag, _xypress
        # print(f"motion event.button: {event.button}")
        for a, _ind in _xypress:
            # safer to use the recorded button at the _press than current
            # button: # multiple button can get pressed during motion...
            a.drag_pan(1, event.key, event.x, event.y)
        if listener is not None:
            listener(event)
        # fig.canvas.draw_idle()

    enable()
    return disable

import traitlets
@traitlets.signature_has_traits
class AxisTraitlet(traitlets.HasTraits):
    xlim = traitlets.Tuple()
    ylim = traitlets.Tuple()
    zlim = traitlets.Tuple()
    def __init__(self, ax, **kwargs):
        self._ax = ax
        # code duplication is easier now
        self._ax.callbacks.connect('xlim_changed', self._on_ax_xlim_change)
        self._ax.callbacks.connect('ylim_changed', self._on_ax_ylim_change)
        self._ax.callbacks.connect('zlim_changed', self._on_ax_zlim_change)
        self.xlim = self._ax.get_xlim()
        super().__init__(**kwargs)
    # TODO: tests - should check that:
    # after: ax_traits.xlim = (0, 2)
    # or after: ax.set_xlim(0, 3)
    # we have: ax.get_xlim() == ax_traits.xlim
    # and "xlim_changed" wasn't called twice but only once
    # and that xlim observer was called once too
    def _on_ax_xlim_change(self, ax):
        self.xlim = self._ax.get_xlim()
    def _on_ax_ylim_change(self, ax):
        self.ylim = self._ax.get_ylim()
    def _on_ax_zlim_change(self, ax):
        self.zlim = self._ax.get_zlim()

    # changes to this object
    @traitlets.observe("xlim")
    def _on_xlim_change(self, event):
        if event['type'] == 'change':
            if self._ax.get_xlim() != event['new']:
                self._ax.set_xlim(event['new'])
    @traitlets.observe("ylim")
    def _on_ylim_change(self, event):
        if event['type'] == 'change':
            if self._ax.get_ylim() != event['new']:
                self._ax.set_ylim(event['new'])
    @traitlets.observe("zlim")
    def _on_zlim_change(self, event):
        if event['type'] == 'change':
            if self._ax.get_zlim() != event['new']:
                self._ax.set_zlim(event['new'])

            
        


from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import re
import types

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook#, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

def _process_plot_format(fmt, *, ambiguous_fmt_datakey=False):
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

    # First check whether fmt is just a colorspec, but specifically exclude the
    # grayscale string "1" (not "1.0"), which is interpreted as the tri_down
    # marker "1".  The grayscale string "0" could be unambiguously understood
    # as a color (black) but also excluded for consistency.
    if fmt not in ["0", "1"]:
        try:
            color = mcolors.to_rgba(fmt)
            return linestyle, marker, color
        except ValueError:
            pass

    errfmt = ("{!r} is neither a data key nor a valid format string ({})"
              if ambiguous_fmt_datakey else
              "{!r} is not a valid format string ({})")

    i = 0
    while i < len(fmt):
        c = fmt[i]
        if fmt[i:i+2] in mlines.lineStyles:  # First, the two-char styles.
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            linestyle = fmt[i:i+2]
            i += 2
        elif c in mlines.lineStyles:
            if linestyle is not None:
                raise ValueError(errfmt.format(fmt, "two linestyle symbols"))
            linestyle = c
            i += 1
        elif c in mlines.lineMarkers:
            if marker is not None:
                raise ValueError(errfmt.format(fmt, "two marker symbols"))
            marker = c
            i += 1
        elif c in mcolors.get_named_colors_mapping():
            if color is not None:
                raise ValueError(errfmt.format(fmt, "two color symbols"))
            color = c
            i += 1
        elif c == "C":
            cn_color = re.match(r"C\d+", fmt[i:])
            if not cn_color:
                raise ValueError(errfmt.format(fmt, "'C' must be followed by a number"))
            color = mcolors.to_rgba(cn_color[0])
            i += len(cn_color[0])
        else:
            raise ValueError(errfmt.format(fmt, f"unrecognized character {c!r}"))

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

    def __init__(self, command='plot'):
        self.command = command
        self.set_prop_cycle(None)

    def set_prop_cycle(self, cycler):
        if cycler is None:
            cycler = mpl.rcParams['axes.prop_cycle']
        self._idx = 0
        self._cycler_items = [*cycler]

    def __call__(self, axes, *args, data=None, **kwargs):
        axes._process_unit_info(kwargs=kwargs)

        for pos_only in "xy":
            if pos_only in kwargs:
                raise _api.kwarg_error(self.command, pos_only)

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
        ambiguous_fmt_datakey = data is not None and len(args) == 2

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
            yield from self._plot_args(
                axes, this, kwargs, ambiguous_fmt_datakey=ambiguous_fmt_datakey)

    def get_next_color(self):
        """Return the next color in the cycle."""
        entry = self._cycler_items[self._idx]
        if "color" in entry:
            self._idx = (self._idx + 1) % len(self._cycler_items)  # Advance cycler.
            return entry["color"]
        else:
            return "k"

    def _getdefaults(self, kw, ignore=frozenset()):
        """
        If some keys in the property cycle (excluding those in the set
        *ignore*) are absent or set to None in the dict *kw*, return a copy
        of the next entry in the property cycle, excluding keys in *ignore*.
        Otherwise, don't advance the property cycle, and return an empty dict.
        """
        defaults = self._cycler_items[self._idx]
        if any(kw.get(k, None) is None for k in {*defaults} - ignore):
            self._idx = (self._idx + 1) % len(self._cycler_items)  # Advance cycler.
            # Return a new dict to avoid exposing _cycler_items entries to mutation.
            return {k: v for k, v in defaults.items() if k not in ignore}
        else:
            return {}

    def _setdefaults(self, defaults, kw):
        """
        Add to the dict *kw* the entries in the dict *default* that are absent
        or set to None in *kw*.
        """
        for k in defaults:
            if kw.get(k, None) is None:
                kw[k] = defaults[k]

    def _makeline(self, axes, x, y, kw, kwargs):
        kw = {**kw, **kwargs}  # Don't modify the original kw.
        self._setdefaults(self._getdefaults(kw), kw)
        seg = mlines.Line2D(x, y, **kw)
        return seg, kw

    def _makefill(self, axes, x, y, kw, kwargs):
        # Polygon doesn't directly support unitized inputs.
        x = axes.convert_xunits(x)
        y = axes.convert_yunits(y)

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
        ignores = ({'marker', 'markersize', 'markeredgecolor',
                    'markerfacecolor', 'markeredgewidth'}
                   # Also ignore anything provided by *kwargs*.
                   | {k for k, v in kwargs.items() if v is not None})

        # Only using the first dictionary to use as basis
        # for getting defaults for back-compat reasons.
        # Doing it with both seems to mess things up in
        # various places (probably due to logic bugs elsewhere).
        default_dict = self._getdefaults(kw, ignores)
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

    def _plot_args(self, axes, tup, kwargs, *,
                   return_kwargs=False, ambiguous_fmt_datakey=False):
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
            Whether to also return the effective keyword arguments after label
            unpacking as well.

        ambiguous_fmt_datakey : bool
            Whether the format string in *tup* could also have been a
            misspelled data key.

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
            linestyle, marker, color = _process_plot_format(
                fmt, ambiguous_fmt_datakey=ambiguous_fmt_datakey)
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

        if axes.xaxis is not None:
            axes.xaxis.update_units(x)
        if axes.yaxis is not None:
            axes.yaxis.update_units(y)

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

        if cbook.is_scalar_or_string(label):
            labels = [label] * n_datasets
        elif len(label) == n_datasets:
            labels = label
        elif n_datasets == 1:
            msg = (f'Passing label as a length {len(label)} sequence when '
                    'plotting a single dataset is deprecated in Matplotlib 3.9 '
                    'and will error in 3.11.  To keep the current behavior, '
                    'cast the sequence to string before passing.')
            _api.warn_deprecated('3.9', message=msg)
            labels = [label]
        else:
            raise ValueError(
                f"label must be scalar or have the same length as the input "
                f"data, but found {len(label)} for {n_datasets} datasets.")

        result = (make_artist(axes, x[:, j % ncx], y[:, j % ncy], kw,
                              {**kwargs, 'label': label})
                  for j, label in enumerate(labels))

        if return_kwargs:
            return list(result)
        else:
            return [l[0] for l in result]



# https://github.com/matplotlib/matplotlib/blob/a254b687df97cda8c6affa37a1dfcf213f8e6c3a/lib/matplotlib/axes/_base.py#L1257
def __clear(self):
        """Clear the Axes."""
        # The actual implementation of clear() as long as clear() has to be
        # an adapter delegating to the correct implementation.
        # The implementation can move back into clear() when the
        # deprecation on cla() subclassing expires.

        # stash the current visibility state
        if hasattr(self, 'patch'):
            patch_visible = self.patch.get_visible()
        else:
            patch_visible = True

        xaxis_visible = self.xaxis.get_visible()
        yaxis_visible = self.yaxis.get_visible()

        for axis in self._axis_map.values():
            axis.clear()  # Also resets the scale to linear.
        for spine in self.spines.values():
            spine._clear()  # Use _clear to not clear Axis again

        self.ignore_existing_data_limits = True
        # self.callbacks = cbook.CallbackRegistry(
        #     signals=["xlim_changed", "ylim_changed", "zlim_changed"])

        # update the minor locator for x and y axis based on rcParams
        if mpl.rcParams['xtick.minor.visible']:
            self.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        if mpl.rcParams['ytick.minor.visible']:
            self.yaxis.set_minor_locator(mticker.AutoMinorLocator())

        self._xmargin = mpl.rcParams['axes.xmargin']
        self._ymargin = mpl.rcParams['axes.ymargin']
        self._tight = None
        self._use_sticky_edges = True

        self._get_lines = _process_plot_var_args()
        self._get_patches_for_fill = _process_plot_var_args('fill')

        self._gridOn = mpl.rcParams['axes.grid']
        old_children, self._children = self._children, []
        for chld in old_children:
            chld.axes = chld.figure = None
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

        if self._sharex is not None:
            self.xaxis.set_visible(xaxis_visible)
            self.patch.set_visible(patch_visible)
        if self._sharey is not None:
            self.yaxis.set_visible(yaxis_visible)
            self.patch.set_visible(patch_visible)

        # This comes last, as the call to _set_lim may trigger an autoscale (in
        # case of shared axes), requiring children to be already set up.
        for name, axis in self._axis_map.items():
            share = getattr(self, f"_share{name}")
            if share is not None:
                getattr(self, f"share{name}")(share)
            else:
                # Although the scale was set to linear as part of clear,
                # polar requires that _set_scale is called again
                if self.name == "polar":
                    axis._set_scale("linear")
                # axis._set_lim(0, 1, auto=True) #!!!! This is not needed
        self._update_transScale()

        self.stale = True