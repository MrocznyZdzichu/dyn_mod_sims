import matplotlib.pyplot as plt
import numpy as np

# Focus on SISO objects for now
def plot_regulation_history(obj, control_mapping, figsize=None):
    fig, axes = _initialize_figure(control_mapping, figsize)
    controlled_outputs = list(control_mapping.keys())
    timestamps = _get_timestamps(obj)

    curr_serie = 0
    tunings_mapping = {}
    
    for n in controlled_outputs:
        pid = control_mapping[n][1]
        bounds = _get_control_bounds(pid)
        
        y  = _get_output(obj, n)
        sp = _get_setpoints(pid)
        cv = _get_control_signal(pid)
        d  = _get_disturbances(obj, control_mapping)
        
        _plot_output(axes[0][curr_serie], timestamps, y)
        _plot_setpoint(axes[0][curr_serie], timestamps, sp)
        _plot_cv(axes[1][curr_serie], timestamps, cv, bounds)
        _plot_disturbances(axes[1][curr_serie], timestamps, d)
        
        curr_serie += 1

    _postprocess_axes(axes, control_mapping)
    _postprocess_figure(fig)
    
def _initialize_figure(control_mapping, figsize=None):
    fig, axes = plt.subplots(2, len(control_mapping.keys()), figsize=figsize)
    fig.set_facecolor([.07, 0.11, 0.15])

    # unify axes array shape
    if len(axes.shape) == 1:
        axes = axes.reshape((2, 1))
    _axes_darkmode(axes)
    return fig, axes

def _axes_darkmode(axes):
    for ax_row in axes:
        for ax in ax_row:
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
        
def _get_timestamps(obj):
    return obj.get_timestamps()
    
def _get_output(obj, n):
    return [y[n][0] for y in obj.get_output_hist()]

def _plot_output(ax, t, y):
    ax.plot(t, y, lw=3, label='Controlled output')

def _get_setpoints(pid):
    return pid.get_setpoint_hist()
    
def _plot_setpoint(ax, t, sp):
    ax.plot(t, sp, '--', label='Setpoint')

def _get_control_signal(pid):
    return pid.get_output_hist()

def _get_pid_parameters(pid):
    return pid.Kp, pid.Ki, pid.Kd

def _get_control_bounds(pid):
    return pid.get_CV_limit()

def _plot_cv(ax, t, cv, bounds):
    ax.plot(t, cv, lw=3, label='Control signal')
    for it, limit in enumerate(bounds):
        if limit != None:
            bound_label = 'CV lower limit' if it == 0 else 'CV upper limit'
            ax.plot(t, [limit]*len(t), '--', label=bound_label)

def _get_disturbances(obj, control_mapping):
    inputs_indices = list(range(0, obj.get_dimensions()[0]))
    controlled_inputs_indices = [val[0] for val in control_mapping.values()]
    dists_inputs_indices = list(set(inputs_indices) - set(controlled_inputs_indices))

    disturbances = {}
    for index in dists_inputs_indices:
        disturbances[index] = [u[index][0] for u in obj.get_input_hist()]

    return disturbances
    
def _plot_disturbances(ax, t, dists):
    for dist_serie, values in dists.items():
        ax.plot(t, values, label=f'Disturbance {dist_serie}', lw=1)
    
def _postprocess_axes(axes, control_mapping):
    upper_charts = axes[0][:]
    lower_charts = axes[1][:]

    for col, ax in enumerate(upper_charts):
        Kp, Ki, Kd = _get_pid_parameters(control_mapping[col][1])
        _postprocess_upper(ax, Kp, Ki, Kd)

    for ax in lower_charts:
        _postprocess_lower(ax)

def _postprocess_upper(ax, Kp, Ki, Kd):
    ax.set_ylabel('Signal value')
    ax.set_title(f'Control parameters: Kp={Kp}, Ki={Ki}, Kd={Kd}')
    ax.legend()

def _postprocess_lower(ax):
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal value')
    ax.legend()

def _postprocess_figure(fig):
    fig.title = 'Control results'