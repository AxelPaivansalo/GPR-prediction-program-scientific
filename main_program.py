import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import LinearNDInterpolator
from iteration_utilities import deepflatten
import argparse
import additional_methods as add_methods
import data_methods

print('---Design the perfect FoamWood block suited for your needs using AI---\n')

data, columns = data_methods.load_data()

# Set signal and noise variance
signal_variance_all = { x: 0.5 for x in columns }
noise_variance_all = { x: 0 for x in columns }

# Configure parser
parser = argparse.ArgumentParser(description='Scientific GPR program')

parser.add_argument(
    '--input', action='append', choices=columns, default=None,
    help="input variable(s) (default: ['Min grad angle(rad/C)', 'Storage modulus at min grad angle(Pa)'])"
)
parser.add_argument(
    '--output', choices=columns, default='Yield stress(Pa*10^6)',
    help="output variable (default: 'Yield stress(Pa*10^6)')"
)
parser.add_argument(
    '--optimize-length-scales-automatically', action='store_false',
    help='use the built-in optimizer to determine length scales (default: False) (warning: turning this feature on usually yields bad results)'
)
parser.add_argument(
    '--length-scales', action='append', default=None, type=float,
    help='length scale(s) of the input variable(s) (default: inferred)'
)
parser.add_argument(
    '--input-values', action='append', default=None, type=float,
    help='value(s) of the input variable(s) (default: inferred)'
)
parser.add_argument(
    '--test-size', default=0.1, type=float,
    help='validation set size (default: 0.1)'
)
parser.add_argument(
    '--seed', default=0, type=int,
    help='seed for random shuffling (default: 0)'
)

args = parser.parse_args()
config = vars(args)

input, output, fix_length_scales, test_size, seed = config['input'], config['output'], config['optimize_length_scales_automatically'], config['test_size'], config['seed']

# Set default value for input arguments
if input == None:
    input = ['Min grad angle(rad/C)', 'Storage modulus at min grad angle(Pa)']

# Sanitize arguments
if len(input) > 3:
    raise ValueError('number of input variables must be 3 or below')

# Create DataFrame-objects
X_data = pd.DataFrame({ x: y for x, y in zip(columns, data) })
Y_data = pd.DataFrame({ x: y for x, y in zip(columns, data) })

X_dropped_col = [ x for x in X_data.columns if x not in input ]
Y_dropped_col = [ x for x in Y_data.columns if x not in output ]

X_data = X_data.drop(columns=X_dropped_col)
Y_data = Y_data.drop(columns=Y_dropped_col)

# Infer length scales and input values
# For further reading on why the length scale bounds are defined like this:
# https://stats.stackexchange.com/questions/297673/how-to-pick-length-scale-bounds-for-rbc-kernels-in-gaussian-process-regression
length_scale_min = [ add_methods.min_dist_arr(X_data[x]) for x in X_data.columns ]
length_scale_max = [ add_methods.max_dist_arr(X_data[x]) for x in X_data.columns ]
length_scales = [ round((x + y) / 2, 2) for x, y in zip(length_scale_max, length_scale_min) ]
length_scale_bounds = [ (x, y) for x, y in zip(length_scale_min, length_scale_max) ]

input_min = [ X_data[x].min() - ((X_data[x].max() - X_data[x].min()) / 10) for x in X_data.columns ]
input_max = [ X_data[x].max() + ((X_data[x].max() - X_data[x].min()) / 10) for x in X_data.columns ]
input_values = [ round((x + y) / 2, 2) for x, y in zip(input_max, input_min) ]

# Override length scales and input values if manual arguments have been given
if config['length_scales'] != None:
    # Sanitize arguments
    if len(config['length_scales']) != len(input):
        raise ValueError('number of length scale arguments must match the number of input variables')

    length_scales = config['length_scales']
if config['input_values'] != None:
    # Sanitize arguments
    if len(config['input_values']) != len(input):
        raise ValueError('number of input value arguments must match the number of input variables')

    input_values = config['input_values']

# Configure signal variance and noise variance
signal_variance = signal_variance_all[Y_data.columns[0]]
noise_variance = noise_variance_all[Y_data.columns[0]]

# Initialize kernel and GP regressor
if fix_length_scales:
    length_scale_bounds = 'fixed'

rbf = ConstantKernel(constant_value=signal_variance) * \
RBF(length_scale=length_scales, length_scale_bounds=length_scale_bounds) + \
WhiteKernel(noise_level=noise_variance)

gpr = GaussianProcessRegressor(kernel=rbf, alpha=0.0)

# Split the data
X_train, X_val, Y_train, Y_val = add_methods.split_data(X_data, Y_data, test_size, seed)

# Create GP model
gpr.fit(X_train, Y_train)

if len(X_data.columns) == 1:
    fig = go.Figure()

    # Create grid
    x_val = np.linspace(
        X_train[X_data.columns[0]].min() - ((X_train[X_data.columns[0]].max() - X_train[X_data.columns[0]].min()) / 10),
        X_train[X_data.columns[0]].max() + ((X_train[X_data.columns[0]].max() - X_train[X_data.columns[0]].min()) / 10),
        100
    )
    X_pred = x_val.reshape(-1, 1)

    # Interpolate
    # Color: https://waldyrious.net/viridis-palette-generator/
    mu_val, std_val = gpr.predict(X_pred, return_std=True)

    fig.add_trace(go.Scatter(
        x=x_val, y=mu_val - std_val, line_width=0,
        line_color='rgba(33, 145, 140, 1)', name=''
    ))
    fig.add_trace(go.Scatter(
        x=x_val, y=mu_val + std_val, fill='tonexty', line_width=0,
        line_color='rgba(33, 145, 140, 1)', name='Standard deviation'
    ))
    fig.add_trace(go.Scatter(
        x=x_val, y=mu_val,
        line_color='rgba(33, 145, 140, 1)', name='Interpolation function'
    ))
    fig.add_trace(go.Scatter(
        x=X_train[X_train.columns[0]], y=Y_train[Y_train.columns[0]], customdata=X_train.index.tolist(),
        line_color='rgba(16, 24, 32, 0.7)', mode='markers', name='Training data'
    ))
    fig.update_layout(
        title='Interpolation of the training data set',
        xaxis_title=X_data.columns[0],
        yaxis_title=Y_data.columns[0]
    )

    # Calculate output value
    X_input = pd.DataFrame({X_data.columns[0]: [input_values[0]]})
    output_value = gpr.predict(X_input)[0]
elif len(X_data.columns) == 2:
    # Create grid
    n = 100
    x_val, y_val = [ np.linspace(
        X_train[x].min() - ((X_train[x].max() - X_train[x].min()) / 10),
        X_train[x].max() + ((X_train[x].max() - X_train[x].min()) / 10),
        n
    ) for x in X_data.columns ]
    xx, yy = np.meshgrid(x_val, y_val)
    X_pred = np.c_[xx.ravel(), yy.ravel()]

    # Interpolate
    mu_val = gpr.predict(X_pred)
    mu_val = np.array(mu_val).reshape((n, n))

    fig = go.Figure(data=[
        go.Surface(
            z=mu_val, x=x_val, y=y_val, coloraxis='coloraxis', opacity=0.8, name='Interpolation function'),
        go.Scatter3d(
            x=X_train[X_train.columns[0]], y=X_train[X_train.columns[1]], z=Y_train[Y_train.columns[0]], customdata=X_train.index.tolist(),
            mode='markers', marker={'size': 6, 'color': Y_train[Y_train.columns[0]], 'coloraxis': 'coloraxis'}, name='Training data')
    ])
    
    fig.update_layout(
        title='Interpolation of the training data set',
        scene={
            'xaxis_title': X_data.columns[0],
            'yaxis_title': X_data.columns[1],
            'zaxis_title': Y_data.columns[0],
            'camera': {'eye': {'x': 2, 'y': 2, 'z': 2}}
        },
        coloraxis={'colorbar': {'title': {'text': '{}'.format(Y_data.columns[0])}}, 'colorscale': 'viridis'}
    )

    # Calculate output value
    X_input = pd.DataFrame({ x: y for x, y in zip(X_data.columns, [[input_values[0]], [input_values[1]]]) })
    output_value = gpr.predict(X_input)[0]
elif len(X_data.columns) == 3:
    # Create grid
    n = 10
    x_val, y_val, z_val = [ np.linspace(
        X_train[x].min() - ((X_train[x].max() - X_train[x].min()) / 10),
        X_train[x].max() + ((X_train[x].max() - X_train[x].min()) / 10),
        n
    ) for x in X_data.columns ]
    xx, yy, zz = np.meshgrid(x_val, y_val, z_val)
    X_pred = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Interpolate
    mu_val = gpr.predict(X_pred)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), text=[ 'color: {}'.format(x) for x in mu_val ],
            mode='markers', marker={'size': 4, 'color': mu_val, 'coloraxis': 'coloraxis'}, name='Interpolation function'),
        go.Scatter3d(
            x=X_train[X_train.columns[0]], y=X_train[X_train.columns[1]], z=X_train[X_train.columns[2]], text=[ 'color: {}'.format(x) for x in Y_train[Y_train.columns[0]] ], customdata=X_train.index.tolist(),
            mode='markers', marker={'size': 12, 'color': Y_train[Y_train.columns[0]], 'coloraxis': 'coloraxis'}, name='Training data')
    ])
    
    fig.update_layout(
        title='Interpolation of the training data set',
        scene={
            'xaxis_title': X_data.columns[0],
            'yaxis_title': X_data.columns[1],
            'zaxis_title': X_data.columns[2],
            'camera': {'eye': {'x': 2, 'y': 2, 'z': 2}}
        },
        coloraxis={'colorbar': {'title': {'text': '{}'.format(Y_data.columns[0])}}, 'colorscale': 'viridis'},
        showlegend=False
    )

    # Calculate output value
    X_input = pd.DataFrame({ x: y for x, y in zip(X_data.columns, [[input_values[0]], [input_values[1]], [input_values[2]]]) })
    output_value = gpr.predict(X_input)[0]

# Return output graph
fig.write_html('figures/general_view.html')

# Return input and output values
print('---Specific point view---')
print(*[ 'Input value of {}: {}'.format(x, round(y, 3)) for x, y in zip(X_data.columns, input_values) ], sep='\n')
print('Output value of {}: {}'.format(Y_data.columns[0], round(output_value, 3)))
print('-----------END-----------\n')

# Model error
mu_val_train = gpr.predict(X_train)
mu_val_val, std_val_val = gpr.predict(X_val, return_std=True)

train_error, val_error = add_methods.gp_error(mu_val_train, mu_val_val, std_val_val, X_val, Y_train, Y_val)

# Return model error
print('---Model error---')
print('Relative training error: {}%'.format(round(train_error, 3)))
print('Relative validation error: {}%'.format(round(val_error, 3)))
print('-------END-------\n')

# Draw PCA figure
X_data_pca = pd.DataFrame({ x: y for x, y in zip(columns, data) })
X_data_pca = X_data_pca.drop(columns=Y_data.columns[0])
Y_data_pca = pd.DataFrame({ x: y for x, y in zip(columns, data) })
Y_data_pca = Y_data_pca.drop(columns=[ x for x in columns if not x == Y_data.columns[0] ])

X_data_pca[X_data_pca.columns] = StandardScaler().fit_transform(X_data_pca[X_data_pca.columns])

# PCA with 2 principal components
pca = PCA(n_components=2)
pca.fit(X_data_pca)
X_data_pca_2pc = pca.transform(X_data_pca)

# Create grid
n = [150, 50]
x_val, y_val = [ np.linspace(
    X_data_pca_2pc[:, x].min() - ((X_data_pca_2pc[:, x].max() - X_data_pca_2pc[:, x].min()) / 10),
    X_data_pca_2pc[:, x].max() + ((X_data_pca_2pc[:, x].max() - X_data_pca_2pc[:, x].min()) / 10),
    y
) for x, y in zip(range(0, 2), n) ]
xx, yy = np.meshgrid(x_val, y_val)

# Interpolate
interp = LinearNDInterpolator(X_data_pca_2pc, Y_data_pca[Y_data_pca.columns[0]])
col_interp = interp(xx, yy)
xx = [ [ b for a, b in zip(x, y) if not np.isnan(a) ] for x, y in zip(col_interp, xx) ]
yy = [ [ b for a, b in zip(x, y) if not np.isnan(a) ] for x, y in zip(col_interp, yy) ]
color_interp = [ [ y for y in x if not np.isnan(y) ] for x in col_interp ]
xx = list(deepflatten(xx))
yy = list(deepflatten(yy))
color_interp = list(deepflatten(color_interp))

fig_pca_2pc = go.Figure()

fig_pca_2pc.add_trace(go.Scatter(
    x=xx, y=yy, text=[ 'color: {}'.format(x) for x in color_interp ],
    mode='markers', marker={'size': 4, 'color': color_interp, 'coloraxis': 'coloraxis'}, name='Linear interpolation'
))
fig_pca_2pc.add_trace(go.Scatter(
    x=X_data_pca_2pc[:, 0], y=X_data_pca_2pc[:, 1], text=[ 'color: {}'.format(x) for x in Y_data_pca[Y_data_pca.columns[0]] ],
    mode='markers', marker={'size': 12, 'color': Y_data_pca[Y_data_pca.columns[0]], 'coloraxis': 'coloraxis'}, name='Training data'
))
fig_pca_2pc.update_layout(
    title='PCA with 2 principal components',
    xaxis_title='PC1',
    yaxis_title='PC2',
    xaxis_range=[x_val[0], x_val[-1]],
    yaxis_range=[y_val[0], y_val[-1]],
    coloraxis={'colorbar': {'title': {'text': '{}'.format(Y_data.columns[0])}}, 'colorscale': 'viridis'},
    showlegend=False
)

# Return PCA with 2PC graph
fig_pca_2pc.write_html('figures/pca_2pc.html')

# PCA with 3 principal components
pca = PCA(n_components=3)
pca.fit(X_data_pca)
X_data_pca_3pc = pca.transform(X_data_pca)

# Create grid
n = 10
x_val, y_val, z_val = [ np.linspace(
    X_data_pca_3pc[:, x].min() - ((X_data_pca_3pc[:, x].max() - X_data_pca_3pc[:, x].min()) / 10),
    X_data_pca_3pc[:, x].max() + ((X_data_pca_3pc[:, x].max() - X_data_pca_3pc[:, x].min()) / 10),
    n
) for x in range(0, 3) ]
xx, yy, zz = np.meshgrid(x_val, y_val, z_val)

# Interpolate
interp = LinearNDInterpolator(X_data_pca_3pc, Y_data_pca[Y_data_pca.columns[0]])
col_interp = interp(xx, yy, zz)
xx = [ [ [ j for i, j in zip(a, b) if not np.isnan(i) ] for a, b in zip(x, y) ] for x, y in zip(col_interp, xx) ]
yy = [ [ [ j for i, j in zip(a, b) if not np.isnan(i) ] for a, b in zip(x, y) ] for x, y in zip(col_interp, yy) ]
zz = [ [ [ j for i, j in zip(a, b) if not np.isnan(i) ] for a, b in zip(x, y) ] for x, y in zip(col_interp, zz) ]
color_interp = [ [ [ i for i in a if not np.isnan(i) ] for a in x ] for x in col_interp ]
xx = list(deepflatten(xx))
yy = list(deepflatten(yy))
zz = list(deepflatten(zz))
color_interp = list(deepflatten(color_interp))

fig_pca_3pc = go.Figure(data=[
    go.Scatter3d(
        x=xx, y=yy, z=zz, text=[ 'color: {}'.format(x) for x in color_interp ],
        mode='markers', marker={'size': 4, 'color': color_interp, 'coloraxis': 'coloraxis'}, name='Linear interpolation'),
    go.Scatter3d(
        x=X_data_pca_3pc[:, 0], y=X_data_pca_3pc[:, 1], z=X_data_pca_3pc[:, 2], text=[ 'color: {}'.format(x) for x in Y_data_pca[Y_data_pca.columns[0]] ],
        mode='markers', marker={'size': 12, 'color': Y_data_pca[Y_data_pca.columns[0]], 'coloraxis': 'coloraxis'}, name='Training data')
])

fig_pca_3pc.update_layout(
    title='PCA with 3 principal components',
    scene={
        'xaxis_title': 'PC1',
        'yaxis_title': 'PC2',
        'zaxis_title': 'PC3',
        'xaxis_range': [x_val[0], x_val[-1]],
        'yaxis_range': [y_val[0], y_val[-1]]
    },
    coloraxis={'colorbar': {'title': {'text': '{}'.format(Y_data.columns[0])}}, 'colorscale': 'viridis'},
    showlegend=False
)

# Return PCA with 3PC graph
fig_pca_3pc.write_html('figures/pca_3pc.html')

# PCA table
pca_table_columns = ['Feature', 'Component 1', 'Component 2', 'Component 3']
comp = pca.components_
pca_table_data = [ {
    pca_table_columns[0]: col,
    pca_table_columns[1]: round(pc1, 3),
    pca_table_columns[2]: round(pc2, 3),
    pca_table_columns[3]: round(pc3, 3)
} for col, pc1, pc2, pc3 in zip(X_data_pca.columns, comp[0, :], comp[1, :], comp[2, :]) ]

# Return PCA table
pd.DataFrame(pca_table_data).to_csv('output_tables/pca_table.csv')

# Draw parallel coordinates figure
fig_par_coo = go.Figure(data=
    go.Parcoords(
        line={
            'color': Y_train[Y_train.columns[0]],
            'colorscale': 'viridis',
            'showscale': True
        },
        dimensions=[ {'label': x, 'values': X_train[x]} for x in X_train.columns ] + [{'label': Y_train.columns[0], 'values': Y_train[Y_train.columns[0]]}]
    )
)

# Return parallel coordinates figure
fig_par_coo.write_html('figures/par_coo.html')

# Data set table
data_table_columns = ['Data point'] + columns
XY_data = pd.DataFrame({ x: y for x, y in zip(columns, data) })
data_table_data = [ dict(zip(data_table_columns, [col] + [ round(x, 3) for x in data ])) for col, data in zip(range(0, len(XY_data.index)), zip(*np.transpose(XY_data.to_numpy()).tolist())) ]

# Return data table
pd.DataFrame(data_table_data).to_csv('output_tables/data_table.csv')
