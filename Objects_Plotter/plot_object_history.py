import matplotlib.pyplot as plt


def plot_object_history(object, figsize=None):
    t = _get_object_timestamps(object)
    u = _get_object_input_hist(object)
    y = _get_object_output_hist(object)

    n_inputs, n_outputs = _get_object_dimensions(object)

    fig, axes = _initialize_figure(n_outputs, figsize)
    axes = _handle_single_axes(axes)
    for out in range(0, n_outputs):
        _plot_output(axes[out], out, n_inputs, t, u, y)
        
def _get_object_timestamps(object):
    return object.get_timestamps()

def _get_object_input_hist(object):
    return object.get_input_hist()

def _get_object_output_hist(object):
    return object.get_output_hist()

def _get_object_dimensions(object):
    n_inputs, n_outputs, n_states = object.get_dimensions()
    return n_inputs, n_outputs

def _initialize_figure(n_outputs, figsize=None):
    fig, axes = plt.subplots(n_outputs, 1, figsize=figsize)
    fig.set_facecolor([.07, 0.11, 0.15])
    return fig, axes

def _handle_single_axes(axes):
    try:
        axes[0]
    except:
        axes = [axes]
    return axes
    
def _plot_output(ax, output, n_inputs, timestamps, inputs, outputs):
    for inp in range(0, n_inputs):
        u_serie = [u[inp][0] for u in inputs]
        ax.plot(timestamps, u_serie, '--', label=f'Input signal {inp+1}')

    y_serie = [y[output][0] for y in outputs]
    ax.plot(timestamps, y_serie, label=f'Output signal {output+1}')
    
    ax.set_title(f'System response {output+1}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal value')
    _ax_darkmode(ax)
    sink = ax.legend()

def _ax_darkmode(ax):
    ax.set_facecolor([.07, 0.11, 0.15])
    ax.spines['bottom'].set_color('#ffffff')
    ax.spines['top'].set_color('#ffffff') 
    ax.spines['right'].set_color('#ffffff')
    ax.spines['left'].set_color('#ffffff')
    ax.tick_params(axis='x', colors='#ffffff')
    ax.tick_params(axis='y', colors='#ffffff')
    ax.yaxis.label.set_color('#ffffff')
    ax.xaxis.label.set_color('#ffffff')
    ax.title.set_color('#ffffff')
