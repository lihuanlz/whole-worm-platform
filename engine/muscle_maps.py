# muscle_maps.py -- neuron->muscle innervation table loader.
# Table provenance: BAAIWorm (BAAI) eworm/components/param/connection/neuron_muscle.xlsx,
# re-exported as data/neuron_muscle.csv (96 body-wall muscle cells x motor neurons).
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "neuron_muscle.csv"


def load_muscle_maps(net):
    import csv
    with open(DATA, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    muscles = rows[0][1:]
    nmn = [r[0] for r in rows[1:]]
    NM = np.nan_to_num(np.array(
        [[float(x) if x else np.nan for x in r[1:]] for r in rows[1:]], dtype=float))
    wout = np.zeros((net.N, 96)); inhib = np.zeros(net.N)
    for r, n in enumerate(nmn):
        if n in net.idx:
            wout[net.idx[n]] = NM[r]
            if n.startswith(("DD", "VD")):
                inhib[net.idx[n]] = 1.0
    n_inn = np.maximum(wout.sum(0), 1.0)
    return muscles, wout, inhib, n_inn
