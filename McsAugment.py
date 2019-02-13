import numpy as np

from scikits.optimization.optimizer.mcs import MCS
from scikits.optimization.optimizer.mcs import Rosenbrock
from scikits.optimization.optimizer.mcs import criterion


if __name__ == "__main__":
  from numpy.testing import *
  startPoint = np.array((-1.01, 1.01), np.float)
  u = np.array((-2.0, -2.0), np.float)
  v = np.array((2.0, 2.0), np.float)

  optimi = MCS(function=Rosenbrock(), criterion=criterion.OrComposition(criterion.MonotonyCriterion(0.00001), criterion.IterationCriterion(10000)), x0=startPoint, u=u, v=v)
  print(dir(optimi))
  print(optimi.optimal_values)
  assert_almost_equal(optimi.optimize(), np.ones(2, np.float), decimal=1)