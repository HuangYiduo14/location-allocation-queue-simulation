import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
# exp1. validate error

def exp1_results_rho(data_file_name):
    data = pd.read_csv(data_file_name)
    data['approximation error'] = (data['simulation sojourn time'] - data['approximated sojourn time'])/data['approximated sojourn time']
    data['abs approximation error'] = data['approximation error'].abs()
    data['approximation error from kingman'] = (data['simulation sojourn time'] - data['Kingman formula sojourn time'])/data['approximated sojourn time']
    data['abs approximation error from kingman'] = data['approximation error from kingman'].abs()
    plt.plot(data['rho2'], data['abs approximation error'], label='absolute relative error vs simulation')
    plt.plot(data['rho2'], data['abs approximation error from kingman'], label='absolute relative error vs kingman')
    plt.plot(data['rho2'], data['error predict theorem'], label='absolute relative error bound by thorem')
    plt.legend()
    plt.xlabel(r'$\rho_2$')
    plt.ylabel('relative error')
    plt.show()

def exp1_result_capacity(data_file_name):
    data = pd.read_csv(data_file_name)
    data['approximation error'] = (data['simulation sojourn time'] - data['approximated sojourn time']) / data[
        'approximated sojourn time']
    data['abs approximation error'] = data['approximation error'].abs()
    data['approximation error from kingman'] = (data['simulation sojourn time'] - data[
        'Kingman formula sojourn time']) / data['approximated sojourn time']
    data['abs approximation error from kingman'] = data['approximation error from kingman'].abs()
    plt.scatter(data['special pod capacity'], data['abs approximation error'], label='absolute relative error vs simulation')
    plt.scatter(data['special pod capacity'], data['abs approximation error from kingman'], label='absolute relative error vs kingman')
    plt.plot(data['special pod capacity'], data['error_predict_theorem'], label='absolute relative error bound by thorem')
    plt.legend()
    plt.xlabel(r'$\xi$')
    plt.ylabel('relative error')
    plt.show()

#exp1_results_rho('rho_result.csv')
exp1_result_capacity('capacity_result.csv')