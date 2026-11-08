"""Triple-existence index: does this endpoint actually HAVE this triple?

The dump is the ground truth for executability.  A federated query whose
SERVICE block matches zero triples eliminates the whole solution, so a
candidate combination that cannot fire is not a slightly-wrong answer --
it is a dead query.

This module answers the question directly: given an endpoint, a subject URI
and a predicate, does the (subject, predicate) pair occur in that
endpoint's dump?

WHAT IS AND IS NOT IN THE INDEX
-------------------------------
The index is built from the partitioned dumps (_split/<kg>/classes/*.nt plus
untyped.nt), which hold the DATA triples.  Label and title triples are
routed to a separate file during partitioning, so rdfs:label and
dcterms:title are NOT in this index.  That is fine here: the index is used
to check CONTENT predicates, and the name triple is instead verified
against the entity index, which only ever returns entities that carry a
name.

STORAGE
-------
35.7M dbpedia triples are far too many to hold as Python strings on an
8 GB box, so each (subject, predicate) pair is hashed to two uint64s and
stored as a sorted structured array, memmapped for lookup.
"""
import hashlib
import os

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
TRIPLE_DIR = os.path.join(BASE, '_triple_index')

DT = np.dtype([('s', '<u8'), ('p', '<u8')])

# The three endpoints, in the order the paper reports them.
KGS = ('dbpedia', 'lmdb', 'drugbank')


def h64(s):
    return int.from_bytes(
        hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'big')


class TripleIndex:
    """(subject, predicate) existence, per endpoint."""

    def __init__(self, d=TRIPLE_DIR):
        self.arr = {}
        for kg in KGS:
            p = os.path.join(d, '%s.npy' % kg)
            self.arr[kg] = np.load(p, mmap_mode='r') if os.path.exists(p) else None

    def has(self, kg, subject_uri, pred_uri):
        """Does this endpoint contain the (subject, predicate) pair?

        Returns None when the endpoint's index is missing, so a caller can
        tell "not present" from "not checked".
        """
        a = self.arr.get(kg)
        if a is None:
            return None
        key = np.array((h64(subject_uri), h64(pred_uri)), dtype=DT)
        i = int(np.searchsorted(a, key))
        return bool(i < len(a) and a[i] == key)

    def any_predicate(self, kg, subject_uri, preds):
        """Does the subject have ANY of these predicates?"""
        for p in preds:
            if self.has(kg, subject_uri, p):
                return True
        return False
