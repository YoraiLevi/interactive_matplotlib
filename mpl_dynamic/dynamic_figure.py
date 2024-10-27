from .matplotlib.matplotlib39_cla import __clear
# from .matplotlib.matplotlib35_cla import cla

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