import matplotlib
from matplotlib import pyplot as plt
import mlrose_hiive as mlrose
from mlrose_hiive import simulated_annealing as sa, random_hill_climb as rhc, genetic_alg as ga, mimic
from mlrose_hiive import RHCRunner, SARunner, GARunner, MIMICRunner, NNGSRunner, ExpDecay, GeomDecay, FourPeaks, FlipFlop, SixPeaks
from mlrose_hiive.algorithms import gradient_descent as gd
import numpy as np
import pandas as pd
import random
import os
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

RS = 199101440
np.random.seed(RS) # keep results consistent
random.seed(RS)
smote = SMOTE(random_state=RS)

def queens_problem(n=8):
    queens_max = lambda state : sum(np.arange(len(state))) - mlrose.Queens().evaluate(state)
    fitness_queens = mlrose.CustomFitness(queens_max)
    return mlrose.DiscreteOpt(length=n,fitness_fn=fitness_queens,maximize = True,max_val=n)

def get_problem(name, n):
    if name == "queens":
        problem = queens_problem(n)
    elif name == "four_peaks":
        problem = mlrose.DiscreteOpt(length=n, fitness_fn=FourPeaks(), maximize=True, max_val=2)
    elif name == "flip_flop":
        problem = mlrose.DiscreteOpt(length=n, fitness_fn=FlipFlop(), maximize=True, max_val=2)
    elif name == "six_peaks":
        problem =  mlrose.DiscreteOpt(length=n, fitness_fn=SixPeaks(), maximize=True, max_val=2)
    else:
        raise Exception("Invalid Problem Name!")
    problem.set_mimic_fast_mode(True)
    return problem

def get_optimal_values(name, n):
    if name == "queens":
        return np.arange(n).sum()
    elif name  == "four_peaks":
        fitness = mlrose.FourPeaks()
        state = np.zeros(n, dtype=int)
        t = int(np.ceil(fitness.t_pct*n))
        state[0:t+1] = 1
        return fitness.evaluate(state)
    elif name == "flip_flop":
        return n - 1
    elif name  == "six_peaks":
        fitness = mlrose.SixPeaks()
        state = np.zeros(n, dtype=int)
        t = int(np.ceil(fitness.t_pct*n))
        state[0:t+1] = 1
        return fitness.evaluate(state)
    else:
        raise Exception("Invalid Problem Name!")

def run_problem_config(config, OUTPUT_FOLDER="output"):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if config['name'] == "nn":
        run_nn_problem_config(config, OUTPUT_FOLDER)
        return

    rhc_run_stats, rhc_run_curves, sa_run_stats, sa_run_curves, ga_run_stats, ga_run_curves, mimic_run_stats, mimic_run_curves = np.full(8, None)

    problem = get_problem(config['name'], config['size'])
    name = config['name']
    size = config['size']
    rhc_runner = RHCRunner(problem, experiment_name=f"{size}-{name}-rhc", seed=RS, **config['rhc'])
    sa_runner = SARunner(problem,experiment_name=f"{size}-{name}-sa",seed=RS,decay_list=[GeomDecay], **config['sa'])
    ga_runner = GARunner(problem,experiment_name=f'{size}-{name}-ga',seed=RS, **config['ga'])
    mimic_runner = MIMICRunner(problem=problem,experiment_name=f"{size}-{name}-mimic",seed=RS,use_fast_mimic=True, **config['mimic'])

    problem_size_results = {"optimal": [get_optimal_values(name, n) for n in config['sizes']]}
    times_results = {}

    if not config['rhc']['skip']:
        print(f"generating rhc results for {size}-{name}...")
        rhc_run_stats, rhc_run_curves = rhc_runner.run()
        rrs = rhc_run_stats[rhc_run_stats['Iteration'] != 0]
        best_restarts = [rrs[rrs['Restarts'].eq(i)]['Fitness'].idxmax() for i in config['rhc']['restart_list']]
        rrs = rrs[rrs.index.isin(best_restarts)]
        rrs.reset_index(inplace=True, drop=True)
        rhc_run_stats = rrs
        best_run = rrs.query('Fitness == Fitness.max()').query('Restarts == Restarts.max()').iloc[0]
        best_restarts = int(best_run['Restarts'])
        rhc_run_curves = rhc_run_curves.query(f"(current_restart == {best_run['current_restart']}) & (Restarts == {best_run['Restarts']})")
        rhc_run_curves.reset_index(inplace=True, drop=True)
        times_results['RHC'] = best_run['Time']
        print(f"best restarts: {best_restarts}\tTime: {best_run['Time']}\tIterations:{rhc_run_curves['Iteration'].iloc[-1]}\n")

        fig, axes = plt.subplots(nrows=1, ncols=2)
        rhc_run_curves.plot(title="RHC Convergence", xlabel="Iterations", ylabel="Fitness", ax=axes[0], x="Iteration", y="Fitness")
        rhc_run_stats.plot(title="RHC Restarts", x="Restarts", y="Fitness", ax=axes[1], marker='o',linestyle='--', xlabel="Number of Restarts", ylabel="Fitness")
        fig.suptitle(f"{size}-{name} RHC")
        fig.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/{size}-{name}-rhc_charts.png")

        max_iters = max(config['rhc']['iteration_list'])
        max_attempts = config['rhc']['max_attempts']
        rhc_problem_results = np.zeros(len(config['sizes']))
        for i in range(len(config['sizes'])):
            n = config['sizes'][i]
            start_time = time.time()
            rhc_problem_results[i] = rhc(get_problem(config['name'], n),max_attempts=max_attempts,max_iters=max_iters,random_state=RS, restarts=best_restarts)[1]
            running_time = time.time() - start_time
            print(f"Completed for size {n}\ttime: {running_time}\tFitness: {rhc_problem_results[i]}")
        problem_size_results["RHC"] = rhc_problem_results
        print()

    if not config['sa']['skip']:
        print(f"generating sa results for {size}-{name}...")
        sa_run_stats, sa_run_curves = sa_runner.run()
        sars = sa_run_stats[sa_run_stats['Iteration'] != 0]
        sars.reset_index(inplace=True, drop=True)
        sa_run_stats = sars
        best_run = sars.query('Fitness == Fitness.max()').iloc[0]
        best_temp = best_run['Temperature']
        sa_run_curves = sa_run_curves[sa_run_curves['Temperature'] == best_run['Temperature']]
        sa_run_curves.reset_index(inplace=True, drop=True)
        times_results['SA'] = best_run['Time']
        print(f"best Temperature: {best_temp}\tTime: {best_run['Time']}\tIterations:{sa_run_curves['Iteration'].iloc[-1]}\n")

        fig, axes = plt.subplots(nrows=1, ncols=2)
        sa_run_curves.plot(title="SA Convergence", xlabel="Iterations", ylabel="Fitness", ax=axes[0], x="Iteration", y="Fitness")
        sa_run_stats.plot(title="SA Initial Temperature", x="schedule_init_temp", y="Fitness", ax=axes[1], marker='o',linestyle='--', xlabel="Temperature", ylabel="Fitness")
        fig.suptitle(f"{size}-{name} SA")
        fig.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/{size}-{name}-sa_charts.png")

        max_iters = max(config['sa']['iteration_list'])
        max_attempts = config['sa']['max_attempts']
        sa_problem_results = np.zeros(len(config['sizes']))
        for i in range(len(config['sizes'])):
            n = config['sizes'][i]
            start_time = time.time()
            sa_problem_results[i] = sa(get_problem(config['name'], n),max_attempts=max_attempts,max_iters=max_iters,random_state=RS, schedule=GeomDecay(init_temp=best_temp.init_temp))[1]
            running_time = time.time() - start_time
            print(f"Completed for size {n}\ttime: {running_time}\tFitness: {sa_problem_results[i]}")
        problem_size_results["SA"] = sa_problem_results
        print()


    if not config['ga']['skip']:
        print(f"generating ga results for {size}-{name}...")
        ga_run_stats, ga_run_curves = ga_runner.run()
        ga_run_stats = ga_run_stats[ga_run_stats['Iteration'] != 0]
        pop_size_results = ga_run_stats.groupby('Population Size')['Fitness'].max()
        mut_rate_results = ga_run_stats.groupby('Mutation Rate')['Fitness'].max()
        ga_groups = ga_run_stats.groupby(['Mutation Rate', 'Population Size']).max()['Fitness']
        best_mut_rate, best_pop_size = ga_groups.idxmax()
        ga_run_curves = ga_run_curves.query(f"(`Mutation Rate` == {best_mut_rate}) & (`Population Size` == {best_pop_size})")
        ga_run_stats.reset_index(inplace=True, drop=True)
        ga_run_curves.reset_index(inplace=True, drop=True)
        best_time = ga_run_curves.iloc[-1]['Time']
        print(f'best Mutation Rate: {best_mut_rate}\tbest Population Size: {best_pop_size}\tTime: {best_time}\tIterations:{ga_run_curves["Iteration"].iloc[-1]}\n')

        fig, axes = plt.subplots(nrows=1, ncols=3)
        pop_size_results = ga_run_stats.groupby('Population Size')['Fitness'].max()
        mut_rate_results = ga_run_stats.groupby('Mutation Rate')['Fitness'].max()
        ga_run_curves.plot(title="GA Convergence", xlabel="Iterations", ylabel="Fitness", ax=axes[0], x="Iteration", y="Fitness")
        pop_size_results.plot(title="GA Population Sizes", ax=axes[1], marker='o',linestyle='--', xlabel="Population Size", ylabel="Fitness")
        mut_rate_results.plot(title="GA Mutation Rates", ax=axes[2], marker='o',linestyle='--', xlabel="Mutation Rate", ylabel="Fitness")
        fig.suptitle(f"{size}-{name} GA")
        fig.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/{size}-{name}-ga_charts.png")

        max_iters = max(config['ga']['iteration_list'])
        max_attempts = config['ga']['max_attempts']
        ga_problem_results = np.zeros(len(config['sizes']))
        for i in range(len(config['sizes'])):
            n = config['sizes'][i]
            start_time = time.time()
            ga_problem_results[i] = ga(get_problem(config['name'], n),max_attempts=max_attempts,max_iters=max_iters,random_state=RS, pop_size=int(best_pop_size), mutation_prob=best_mut_rate)[1]
            running_time = time.time() - start_time
            print(f"Completed for size {n}\ttime: {running_time}\tFitness: {ga_problem_results[i]}")
        problem_size_results["GA"] = ga_problem_results
        print()

    if not config['mimic']['skip']:
        print(f"generating mimic results for {size}-{name}...")
        mimic_run_stats, mimic_run_curves = mimic_runner.run()
        mimic_run_stats = mimic_run_stats[mimic_run_stats['Iteration'] != 0]
        pop_size_results = mimic_run_stats.groupby('Population Size')['Fitness'].max()
        keep_percent_results = mimic_run_stats.groupby('Keep Percent')['Fitness'].max()
        mimic_groups = mimic_run_stats.groupby(['Keep Percent', 'Population Size']).max()['Fitness']
        best_keep_percent, best_pop_size = mimic_groups.idxmax()
        mimic_run_curves = mimic_run_curves.query(f"(`Keep Percent` == {best_keep_percent}) & (`Population Size` == {best_pop_size})")
        mimic_run_stats.reset_index(inplace=True, drop=True)
        mimic_run_curves.reset_index(inplace=True, drop=True)
        best_time = mimic_run_curves.iloc[-1]['Time']
        times_results['MIMIC'] = best_time
        print(f'best Keep Percentage: {best_keep_percent}\tbest Population Size: {best_pop_size}\tTime: {best_time}\tIterations:{mimic_run_curves["Iteration"].iloc[-1]}\n')

        fig, axes = plt.subplots(nrows=1, ncols=3)
        pop_size_results = mimic_run_stats.groupby('Population Size')['Fitness'].max()
        keep_percent_results = mimic_run_stats.groupby('Keep Percent')['Fitness'].max()
        mimic_run_curves.plot(title="MIMIC Convergence", xlabel="Iterations", ylabel="Fitness", ax=axes[0], x="Iteration", y="Fitness")
        pop_size_results.plot(title="MIMIC Population Sizes", ax=axes[1], marker='o',linestyle='--', xlabel="Population Size", ylabel="Fitness")
        keep_percent_results.plot(title="MIMIC Keep Percent", ax=axes[2], marker='o',linestyle='--', xlabel="Keep Percent", ylabel="Fitness")
        fig.suptitle(f"{size}-{name} MIMIC")
        fig.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/{size}-{name}-mimic_charts.png")

        max_iters = max(config['mimic']['iteration_list'])
        max_attempts = config['mimic']['max_attempts']
        mimic_problem_results = np.zeros(len(config['sizes']))
        for i in range(len(config['sizes'])):
            n = config['sizes'][i]
            start_time = time.time()
            mimic_problem_results[i] = mimic(get_problem(config['name'], n),max_attempts=max_attempts,max_iters=max_iters,random_state=RS, pop_size=int(best_pop_size), keep_pct=best_keep_percent)[1]
            running_time = time.time() - start_time
            print(f"Completed for size {n}\ttime: {running_time}\tFitness: {mimic_problem_results[i]}")
        problem_size_results["MIMIC"] = mimic_problem_results
        print()

    fig, axes = plt.subplots(nrows=1, ncols=1)
    problem_size_results = pd.DataFrame(problem_size_results, index=config['sizes'])
    problem_size_results.index.rename('Problem Size', inplace=True)
    problem_size_results.plot(ax=axes, title=f"{name} Problem Size Results", marker='o',linestyle='--', xlabel="Problem Size", ylabel="Fitness")
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{name}-problem_sizes.png")
    html = problem_size_results.to_html(index=True)
    with open(f"{OUTPUT_FOLDER}/{name}-problem_sizes.html", 'w') as fp:
        fp.write(html)

    times_results = pd.Series(times_results).to_frame()
    times_results.columns = ['Time (s)']
    html = times_results.to_html(index=True)
    with open(f"{OUTPUT_FOLDER}/{size}-{name}-times-results.html", 'w') as fp:
        fp.write(html)

    print("Completed running algorithms.\n")
    return rhc_run_stats, rhc_run_curves, sa_run_stats, sa_run_curves, ga_run_stats, ga_run_curves, mimic_run_stats, mimic_run_curves

def get_diabetes_data():
    data = pd.read_csv("data/diabetes.csv")
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # 80% training and 20% test
    x_train, y_train = smote.fit_resample(x_train, y_train)
    x_train = RobustScaler().fit_transform(x_train)
    x_test = RobustScaler().fit_transform(x_test)
    return x_train, x_test, y_train, y_test

def run_nn_problem_config(config, OUTPUT_FOLDER="output"):
    x_train, x_test, y_train, y_test = get_diabetes_data()
    shared_params = {
        'learning_rate_init': [0.2],
        'activation': [mlrose.relu],
        'hidden_layer_sizes': [[49]]
    }
    runner_params = {
        "x_train" : x_train,
        "y_train" : y_train,
        "x_test" : x_test,
        "y_test" : y_test,
        "experiment_name": "nn",
        "clip_max": 1,
        "max_attempts": 50,
        "n_jobs": 5,
        "seed": RS,
        "cv": 2
    }

    fitness_results = {}
    time_results = {}
    acc_results = {}
    curves_results = {}

    # gradient dececent
    if not config['gd']['skip']:
        print("Running gd nn...")
        it_list = config['gd']['iteration_list']
        shared_params['max_iters'] = [max(config['gd']['iteration_list'])]
        gd_params = shared_params | {"learning_rate_init": [0.0002]}
        gd_nnr = NNGSRunner(algorithm=gd, grid_search_parameters=gd_params,iteration_list=it_list, **runner_params)

        start_time = time.time()
        run_stats, curves, cv_results, grid_search_cv = gd_nnr.run()
        running_time = time.time() - start_time

        run_stats = run_stats[run_stats['Iteration'] != 0]
        run_stats = run_stats.query("Fitness == Fitness.min()")
        run_stats.reset_index(inplace=True, drop=True)

        curves_results['Gradient Descent'] = curves['Fitness']
        curves.plot(title="GD NN Fitness over Iterations", xlabel="Iterations", ylabel="Fitness", x="Iteration", y="Fitness")
        plt.savefig(f"{OUTPUT_FOLDER}/gd-nn-fitness.png")

        best_fitness = run_stats.iloc[0].Fitness
        fitness_results['Gradient Descent'] = best_fitness
        time_results['Gradient Descent'] = running_time

        acc_score = cv_results['mean_train_score'][0]
        acc_results["Gradient Descent"] = acc_score

        print(f"Fitness: {best_fitness}\tTime: {running_time} seconds\taccuracy: {acc_score}\n")

    # randomized hill climbing
    if not config['rhc']['skip']:
        print("Running rhc nn...")
        rhc_params = shared_params | {"restarts" : config['rhc']['restart_list']}
        rhc_params['max_iters'] = [max(config['rhc']['iteration_list'])]
        it_list = config['rhc']['iteration_list']
        rhc_nnr = NNGSRunner(algorithm=rhc, grid_search_parameters=rhc_params,iteration_list=it_list, **runner_params)

        start_time = time.time()
        run_stats, curves, cv_results, grid_search_cv = rhc_nnr.run()
        running_time = time.time() - start_time

        run_stats = run_stats[run_stats['Iteration'] != 0]
        run_stats = run_stats.query("Fitness == Fitness.min()")
        run_stats.reset_index(inplace=True, drop=True)
        curves = curves.query(f"current_restart == {run_stats['current_restart'][0]}")
        curves.reset_index(inplace=True, drop=True)

        curves_results['RHC'] = curves['Fitness']
        curves.plot(title="RHC NN Fitness over Iterations", xlabel="Iterations", ylabel="Fitness", x="Iteration", y="Fitness")
        plt.savefig(f"{OUTPUT_FOLDER}/rhc-nn-fitness.png")

        best_fitness = run_stats.iloc[0].Fitness
        fitness_results['RHC'] = best_fitness
        time_results['RHC'] = running_time
        restarts = run_stats.iloc[0].restarts

        acc_score = cv_results['mean_train_score'][0]
        acc_results["RHC"] = acc_score

        print(f"Fitness: {best_fitness}\tTime: {running_time} seconds\taccuracy: {acc_score}\trestarts: {restarts}\n")

    # simulated annealing
    if not config['sa']['skip']:
        print("Running sa nn...")
        temp_list = [GeomDecay(init_temp=t, min_temp=0.00000000000000000001, decay=0.01) for t in config['sa']['temperature_list']]
        sa_params = shared_params | {"schedule" : temp_list, "max_iters": [max(config['sa']['iteration_list'])]}
        it_list = config['sa']['iteration_list']
        sa_nnr = NNGSRunner(algorithm=sa, grid_search_parameters=sa_params,iteration_list=it_list, **runner_params)

        start_time = time.time()
        run_stats, curves, cv_results, grid_search_cv = sa_nnr.run()
        running_time = time.time() - start_time

        run_stats = run_stats[run_stats['Iteration'] != 0]
        run_stats = run_stats.query("Fitness == Fitness.min()")
        run_stats.reset_index(inplace=True, drop=True)

        curves_results['SA'] = curves['Fitness']
        curves.plot(title="SA NN Fitness over Iterations", xlabel="Iterations", ylabel="Fitness", x="Iteration", y="Fitness")
        plt.savefig(f"{OUTPUT_FOLDER}/sa-nn-fitness.png")

        best_fitness = run_stats.iloc[0].Fitness
        fitness_results['SA'] = best_fitness
        time_results['SA'] = running_time

        acc_score = cv_results['mean_train_score'][0]
        acc_results["SA"] = acc_score

        temp = run_stats.iloc[0].schedule_init_temp
        print(f"Fitness: {best_fitness}\tTime: {running_time} seconds\taccuracy: {acc_score}\ttemperature: {temp}\n")

    # genetic algorithms
    if not config['ga']['skip']:
        print("Running ga nn...")
        ga_params = {"mutation_prob" : config['ga']['mutation_rates'], "pop_size": config['ga']['population_sizes']}
        ga_params = shared_params | ga_params
        ga_params['max_iters'] = [max(config['rhc']['iteration_list'])]
        it_list = config['ga']['iteration_list']
        ga_nnr = NNGSRunner(algorithm=ga, grid_search_parameters=ga_params,iteration_list=it_list, **runner_params)

        start_time = time.time()
        run_stats, curves, cv_results, grid_search_cv = ga_nnr.run()
        running_time = time.time() - start_time

        run_stats = run_stats[run_stats['Iteration'] != 0]
        run_stats = run_stats.query("Fitness == Fitness.min()")
        run_stats.reset_index(inplace=True, drop=True)

        curves_results['GA'] = curves['Fitness']
        curves.plot(title="GA NN Fitness over Iterations", xlabel="Iterations", ylabel="Fitness", x="Iteration", y="Fitness")
        plt.savefig(f"{OUTPUT_FOLDER}/ga-nn-fitness.png")

        best_fitness = run_stats.iloc[0].Fitness
        pop_size = run_stats.iloc[0].pop_size
        mut_rate = run_stats.iloc[0].mutation_prob
        fitness_results['GA'] = best_fitness
        time_results['GA'] = running_time

        acc_score = cv_results['mean_train_score'][0]
        acc_results["GA"] = acc_score

        print(f"Fitness: {best_fitness}\tTime: {running_time} seconds\taccuracy: {acc_score}\tpop_size: {pop_size}\tmut_rate: {mut_rate}\n")

    overall_results = pd.DataFrame([fitness_results, time_results, acc_results])
    overall_results.index = ["Fitness", "Running Time (s)", "accuracy"]

    html = overall_results.to_html(index=True)
    with open(f"{OUTPUT_FOLDER}/nn-results.html", 'w') as fp:
        fp.write(html)
    print(overall_results)
    curves_results = pd.DataFrame(curves_results)
    curves_results.plot(title="NN Convergence: Fitness over Iterations", xlabel="Iterations", ylabel="Fitness")
    plt.savefig(f"{OUTPUT_FOLDER}/nn-fitness.png")
    print()
