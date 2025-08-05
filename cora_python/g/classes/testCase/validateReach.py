import numpy as np
import matplotlib.pyplot as plt
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.contSet import ContSet
import warnings

# Importing ARX classes - assuming they exist at these locations
from cora_python.contDynamics.linearARX.linearARX import LinearARX
from cora_python.contDynamics.nonlinearARX.nonlinearARX import NonlinearARX
from cora_python.contDynamics.nonlinearSysDT.nonlinearSysDT import NonlinearSysDT

def validateReach(self, configs, check_contain=False, plot_settings=None):
    """
    validateReach - evaluate the given test case, i.e., compute the reachable
    set of the systems in configs, check containment of all measurements,
    and plot the results
    """
    if plot_settings is None:
        plot_settings = {}

    R = [None] * len(configs)
    R_p = [None] * len(configs)
    n_k = 0

    eval_results = {
        'num_out': np.zeros(len(configs)),
        'num_in': np.zeros(len(configs)),
        'size_R': np.zeros(len(configs)),
    }

    for i, config in enumerate(configs):
        # 1. compute the reachable set R
        if 'U' not in config['params']:
            R[i] = [None] * self.u.shape[0]
            continue
        
        y = self.y
        params = {'U': config['params']['U']}

        if self.u.shape[2] > 1:
            config['obj'] = self.compute_ya(config['sys'])
            y = config['obj'].y_a
            params['R0'] = config['params']['R0']
            params['u'] = np.zeros((self.u.shape[1], self.u.shape[0]))
        elif isinstance(config['sys'], (LinearARX, NonlinearARX)):
            obj2_list = config['obj'].setInitialStateToMeas(
                config['sys'].p, 1e-9
            )
            if len(obj2_list) > 1:
                raise ValueError("Initial state is a set which is not"
                                 " allowed for ARX models")
            if 'R0' in config['params']:
                if np.sum(np.abs(config['params']['R0'].G)) > 1e-6:
                    raise NotImplementedError("Uncertain R0 not "
                                            "implemented for ARX "
                                            "models.")
                params['R0'] = config['params']['R0']
            else:
                params['R0'] = Zonotope(obj2_list[0].initialState)

            params['obj'] = obj2_list[0]
            params['u'] = self.u.T
        else:
            params['u'] = self.u.T
            params['R0'] = config['params']['R0'] + self.initialState
        
        diff_u = config['sys'].nrOfInputs - params['u'].shape[0]
        if diff_u > 0:
            params['u'] = np.vstack([params['u'], np.zeros((diff_u, params['u'].shape[1]))])

        options = config.get('options', {}).copy()
        if 'cs' in options:
            del options['cs']
            
        n_k = params['u'].shape[1]
        params['tFinal'] = config['sys'].dt * (n_k - 1)
        
        try:
            R_full = config['sys'].reach(params, options)
            R[i] = R_full.timePoint.set
            R_p[i] = R[i]
        except Exception as e:
            warnings.warn(f"Reachability analysis for configuration "
                          f"'{config['name']}' failed: {e}")
            R[i] = [None] * self.u.shape[0]

        if check_contain:
            for k in range(n_k):
                for s in range(y.shape[2]):
                    if not isinstance(R[i][k], ContSet) or R[i][k].isemptyobject() or np.any(np.isnan(y[k,:,s])):
                        continue
                    # Check for empty zonotope or non-containment
                    is_empty_zono = isinstance(R[i][k], Zonotope) and (R[i][k].G is None or np.sum(np.abs(R[i][k].G)) == 0)
                    if (is_empty_zono and np.sum(np.abs(R[i][k].c - y[k,:,s].T)) > 1e-6) or not R[i][k].contains_(y[k,:,s].T, tol=1e-6)[0]:
                        eval_results['num_out'][i] += 1
                    else:
                        eval_results['num_in'][i] += 1
                eval_results['size_R'][i] += np.sum(np.abs(R[i][k].G)) if hasattr(R[i][k], 'G') and R[i][k].G is not None else 0
        
        if isinstance(configs[0].get('sys'), NonlinearSysDT) and isinstance(config.get('sys'), NonlinearARX):
             p = configs[0]['sys'].nrOfDims
             R[i] = R[i][p:]
             y = self.y[p:,:,:]
        
        if plot_settings.get('plot_Yp', False):
            p_GO = config['sys'].computeGO(params['R0'].center(), params['U'].center() + self.u.T, n_k)
            Y_p = [None] * n_k
            for k in range(n_k):
                Y_p[k] = p_GO.y[:,k] + p_GO.C[k] @ (config['params']['R0'] - config['params']['R0'].center())
                for j in range(k + 1):
                    Y_p[k] = Y_p[k] + p_GO.D[k,j] @ (config['params']['U'] - config['params']['U'].center())
            R_p[i] = Y_p

    if not plot_settings or not plot_settings.get('k_plot'):
        return R, eval_results

    # Plotting logic
    lines = plot_settings.get('lines', ["-", "-", "--", "-.", ":"] * (len(configs) // 4 + 1))
    colors = plot_settings.get('colors', plt.cm.get_cmap('tab10').colors)
    linewidths = plot_settings.get('linewidths', [1.5] * len(configs))
    k_plot = plot_settings.get('k_plot', range(1, n_k + 1))
    dims = plot_settings.get('dims', [[0, 1]])
    
    num_row = int(np.ceil(np.sqrt(len(k_plot))))
    num_col = int(np.ceil(len(k_plot) / num_row))

    for i_dim, dim in enumerate(dims):
        plt.figure(figsize=(num_col * 4, num_row * 4))
        if 'name' not in plot_settings:
            plt.suptitle(f'Reachability for y_{dim[0]+1} and y_{dim[1]+1}')
        else:
            plt.suptitle(plot_settings['name'])

        for i_k, k in enumerate(k_plot):
            ax = plt.subplot(num_row, num_col, i_k + 1)
            ax.set_title(f'k = {k}')
            
            for i, config in enumerate(configs):
                name = config.get('name', f'config{i}')
                col = [0.9, 0.9, 0.9] if "true" in name else colors[i % len(colors)]
                alpha = 0.8 if "true" in name else 0.2
                
                name_Y = f"$\\mathcal{{Y}}_{{a, \\mathrm{{{name}}}}}$" if self.u.shape[2] > 1 else f"$\\mathcal{{Y}}_{{\\mathrm{{{name}}}}}$"

                R_ik = R[i][k-1]
                if not isinstance(R_ik, ContSet):
                    R_ik = Zonotope(R_ik)
                
                if R_ik.isemptyobject(): continue

                # Assuming a plot method exists for contSet objects
                R_ik.plot(dims=dim, ax=ax, facecolor=col, edgecolor=col, alpha=alpha, linewidth=linewidths[i], linestyle=lines[i], label=name_Y)
                
                if plot_settings.get('plot_Yp', False) and R_p[i] is not None:
                    R_p_ik = R_p[i][k-1]
                    if R_p_ik is not None:
                         R_p_ik.plot(dims=dim, ax=ax, facecolor='none', edgecolor=col, linewidth=linewidths[i], linestyle='--')
            
            # Plot measurement data
            if 's_val' in plot_settings:
                 y_data = y[k-1, dim, plot_settings['s_val']:]
                 ax.plot(y_data[0,:], y_data[1,:], 'kx')

            ax.legend()
            ax.set_xlabel(f'y_{dim[0]+1}')
            ax.set_ylabel(f'y_{dim[1]+1}')
        
        if 'path_fig' in plot_settings:
            plt.savefig(plot_settings['path_fig'])
        plt.show()

    return R, eval_results
