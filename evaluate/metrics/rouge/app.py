import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("rogue")
launch_gradio_widget(module)
