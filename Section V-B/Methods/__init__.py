from .cagia_naive import run_attack as run_cagia_naive
from .cagia_opt import run_attack as run_cagia_opt
from .dager import run_attack as run_dager
from .dlg import run_attack as run_dlg
from .grab import run_attack as run_grab
from .lamp import run_attack as run_lamp
from .partial_gradient import run_attack as run_partial_gradient
from .tag import run_attack as run_tag


METHODS = {
    "cagia-naive": run_cagia_naive,
    "cagia_naive": run_cagia_naive,
    "cagia-opt": run_cagia_opt,
    "cagia_opt": run_cagia_opt,
    "dager": run_dager,
    "dlg": run_dlg,
    "grab": run_grab,
    "lamp": run_lamp,
    "partial-gradient": run_partial_gradient,
    "partial_gradient": run_partial_gradient,
    "partial": run_partial_gradient,
    "tag": run_tag,
}
