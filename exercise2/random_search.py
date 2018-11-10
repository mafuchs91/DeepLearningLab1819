import logging

logging.basicConfig(level=logging.WARNING)

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import argparse

from cnn_mnist_solution import mnist
from cnn_mnist_solution import train_and_validate
from cnn_mnist_solution import test

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten

# class to keep track of accurancy
class ConvNetHist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc =[]

    def on_epoch_end(self, batch_size, logs={}):
        self.acc.append(logs.get("acc"))


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist("./")

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        filter_size = config["filter_size"]
        epochs = budget
        print(lr, num_filters, batch_size, filter_size)
        # do the actual training
        lc, model = train_and_validate(self.x_train,
                                 self.y_train, self.x_valid, self.y_valid,
                                 epochs, lr, num_filters, filter_size, batch_size)

        # return last element from learning_curve as error and the parameters as info
        return ({
            'loss': lc[-1],
            'info': {lr,num_filters, filter_size, batch_size,epochs}
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        # Add hyperparmeter test intervalls
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter("learning_rate", lower=10**-4, upper=0.1, log=True))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("batch_size", lower=16, upper=128, log=True))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("num_filters", lower=2**3, upper=2**6, log=True))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter("filter_size", [3, 5]))



        # TODO: Implement configuration space here. See https://github.com/automl/HpBandSter/blob/master/hpbandster/examples/example_5_keras_worker.py  for an example

        return config_space


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=6)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=50)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()


# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])


# Plots the performance of the best found validation error over time
all_runs = res.get_all_runs()
# Let's plot the observed losses grouped by budget,
import hpbandster.visualization as hpvis

hpvis.losses_over_time(all_runs)

import matplotlib.pyplot as plt
plt.savefig("random_search.svg")

# train the network ons more with the optimal hyper parameters found in random search
lc, model = train_and_validate(w.x_train,
                                 w.y_train, w.x_valid, w.y_valid, args.budget, id2config[incumbent]["config"]["learning_rate"],
                                  id2config[incumbent]["config"]["num_filters"],id2config[incumbent]["config"]["filter_size"], id2config[incumbent]["config"]["batch_size"])
# store the learning curve
plt.figure(figsize=(13, 10))
plt.xlabel("epoch")
plt.ylabel("error")
plt.plot(range(1,args.budget + 1),lc, label="optimal_params",linestyle='-',color="blue")
plt.legend(loc="best", prop={'size': 15})
plt.savefig('random_search.svg')
# print the test error, from the traind network with the optimal hyperparameters(only 6 epochs of training)
print(test(w.x_test,w.y_test, model))
