"""Custom types used in iNNvestigate"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from keras import Model
from keras.layers import Layer
from tensorflow import Tensor
from typing_extensions import TypedDict

# Define type of checks, using Any for kwargs
LayerCheck = Union[Callable[[Layer], bool], Callable[[Layer, Any], bool]]

# Used for LRP rules
ReverseRule = Tuple[LayerCheck, Any]  # TODO: replace Any with ReverseMappingBase


class ModelCheckDict(TypedDict):
    """ "Adds type hints to model check dicts."""

    check: LayerCheck
    message: str
    check_type: str


class CondReverseMapping(TypedDict):
    """Adds type hints to conditional reverse mapping dicts."""

    condition: LayerCheck
    mapping: Callable  # TODO: specify type
    name: Optional[str]


class NodeDict(TypedDict):
    """Adds type hints to NodeDicts.

    Contains the following items:
    * `nid`: the node id.
    * `layer`: the layer creating this node.
    * `Xs`: the input tensors (only valid if not in a nested container).
    * `Ys`: the output tensors (only valid if not in a nested container).
    * `Xs_nids`: the ids of the nodes creating the Xs.
    * `Ys_nids`: the ids of nodes using the according output tensor.
    * `Xs_layers`: the layer that created the according input tensor.
    * `Ys_layers`: the layers using the according output tensor.

    """

    nid: Optional[int]
    layer: Layer
    Xs: List[Tensor]
    Ys: List[Tensor]
    Xs_nids: List[Optional[int]]
    Ys_nids: List[Union[List[int], List[None]]]
    Xs_layers: List[Layer]
    Ys_layers: List[List[Layer]]
