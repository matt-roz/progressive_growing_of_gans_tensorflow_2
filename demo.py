import scipy.ndimage
import numpy as np
import tensorflow as tf
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column, layout, row
from bokeh.models import Button, TextInput

from layers import PixelNormalization, StandardDeviationLayer, WeightScalingWrapper


# load model
custom = dict((val.__name__, val) for val in [StandardDeviationLayer, WeightScalingWrapper, PixelNormalization])
model = tf.keras.models.load_model(filepath="/media/storage/outs/20200423-183746-deepcera/cp_generator_final_epoch-0378.h5", custom_objects=custom)
model.build(input_shape=(512,))

input_shapes = [val.shape for val in model.inputs]
noise_dim = input_shapes[0][-1]
output_shapes = [val.shape[1:] for val in model.outputs]
print(output_shapes)
max_h, max_w, _ = output_shapes[-1]
ps = [figure(x_range=(0, 1), y_range=(0, 1), plot_height=max_h, plot_width=max_w, match_aspect=False) for height, width, _ in output_shapes]

# p = figure(x_range=(0, 1), y_range=(0, 1), plot_width=128, plot_height=128, match_aspect=True)
for p in ps:
    p.toolbar.logo = None
    p.toolbar_location = None
    p.xaxis.visible = None
    p.yaxis.visible = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

# updating values
noise = tf.random.normal([noise_dim]).numpy()
noise /= np.linalg.norm(noise)

# bokeh elements
text_fps = TextInput(title="uint: Frames per Second")
text_sec = TextInput(title="uint: Duration in Seconds")
btn_randomize = Button(label="Randomize Noise")
btn_rotate = Button(label="► Start Interpolation")
text_fps.value = '10'
text_sec.value = '10'
index = 0
max_index = None
latents = None
callback_id = None


def randomize_noise():
    global index, latents, max_index, noise
    index = 0
    max_index = int(text_fps.value) * int(text_sec.value)
    latents = np.random.randn(max_index, noise_dim)
    latents = scipy.ndimage.gaussian_filter(input=latents, sigma=[int(text_fps.value), 0], mode='wrap')
    latents /= np.linalg.norm(latents)
    noise = latents[index]


def random_interpolation():
    global noise, index, latents, max_index
    index += 1
    noise = latents[index]
    update()
    if index >= max_index - 1:
        stop_animate()
        randomize_noise()


def inference_model():
    results = list(model([np.expand_dims(noise, axis=0), 1.0]))
    res = []
    for result in results:
        result = np.squeeze(result.numpy())
        result = (result + 1.0) / 2.0
        result = np.uint8(result * 255.0)
        alpha = 255 * np.ones(result.shape[:-1] + (1,), dtype=np.uint8)
        result = np.append(result, alpha, axis=-1)
        result[:] = result[::-1]
        res.append(result)
    return res


def update():
    ds = inference_model()
    for d, p in zip(ds, ps):
        p.image_rgba(image=[d], x=[0], y=[0], dw=1, dh=1)


def start_animate():
    global callback_id
    btn_rotate.label = '❚❚ Stop Interpolation'
    callback_id = curdoc().add_periodic_callback(random_interpolation, 1000/int(text_fps.value))


def stop_animate():
    global callback_id
    if btn_rotate.label == '❚❚ Stop Interpolation':
        btn_rotate.label = "► Start Interpolation"
        curdoc().remove_periodic_callback(callback_id)


def animate():
    if btn_rotate.label == "► Start Interpolation":
        start_animate()
    else:
        stop_animate()


# bind actions
text_sec.on_change('value', lambda attr, old, new: randomize_noise())
text_fps.on_change('value', lambda attr, old, new: randomize_noise())
btn_randomize.on_click(randomize_noise)
btn_rotate.on_click(animate)

controls = [text_sec, text_fps, btn_randomize, btn_rotate]

inputs = column(*controls)
inputs.sizing_mode = "fixed"
l = layout([
    [inputs],
    ps[::-1],
], sizing_mode="fixed")

randomize_noise()
update()

curdoc().add_root(l)
curdoc().title = "Movies"
