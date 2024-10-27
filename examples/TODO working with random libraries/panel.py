%pip install pyviz_comms
%pip install https://cdn.holoviz.org/panel/1.5.2/dist/wheels/bokeh-3.6.0-py3-none-any.whl
%pip install https://cdn.holoviz.org/panel/1.5.2/dist/wheels/panel-1.5.2-py3-none-any.whl
%pip install hvplot
%pip install param jupyter_bokeh


import panel as pn
from holoviews.streams import RangeXY
import holoviews as hv
import numpy as np

hv.extension("bokeh")

def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return 255

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image

def get_fractal(x_range, y_range):
    (x0, x1), (y0, y1) = x_range, y_range
    image = np.zeros((600, 600), dtype=np.uint8)
    return hv.Image(create_fractal(x0, x1, -y1, -y0, image, 15),
                    bounds=(x0, y0, x1, y1))
    
range_stream = RangeXY(x_range=(-1., 1.), y_range=(-1., 1.))

dmap = hv.DynamicMap(get_fractal,streams=[range_stream])
dmap_panel = pn.panel(dmap)
dmap_panel