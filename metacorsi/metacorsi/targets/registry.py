from .dsh import DSHAdapter
from .mini_swe_agent import MiniSWEAgentAdapter
from .poa import POAAdapter

registry = {x.name: x for x in (DSHAdapter(), MiniSWEAgentAdapter(), POAAdapter())}
