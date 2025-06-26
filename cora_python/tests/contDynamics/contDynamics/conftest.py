import pytest
from cora_python.contDynamics.contDynamics.contDynamics import ContDynamics

@pytest.fixture
def ConcreteContDynamics():
    class ConcreteContDynamics(ContDynamics):
        def __repr__(self) -> str:
            return (f"ConcreteContDynamics(name='{self.name}', states={self.nr_of_dims}, "
                    f"inputs={self.nr_of_inputs}, outputs={self.nr_of_outputs}, "
                    f"dists={self.nr_of_disturbances}, noises={self.nr_of_noises})")
    return ConcreteContDynamics 