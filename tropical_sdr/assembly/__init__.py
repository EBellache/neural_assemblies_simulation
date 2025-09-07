"""Neural assembly implementation with tropical dynamics."""

from .assembly import TropicalAssembly, AssemblyState
from .competition import CompetitionArena, WinnerTakeAll
from .network import TropicalAssemblyNetwork, NetworkConfig
from .learning import HebbianLearning, LearningConfig

__all__ = [
    'TropicalAssembly',
    'AssemblyState',
    'CompetitionArena', 
    'WinnerTakeAll',
    'TropicalAssemblyNetwork',
    'NetworkConfig',
    'HebbianLearning',
    'LearningConfig'
]