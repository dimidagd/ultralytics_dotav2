import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("map")
launch_gradio_widget(module)
