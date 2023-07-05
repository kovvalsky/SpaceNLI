#!/usr/bin/env python3
# -*- coding: utf8 -*-

import yaml
from collections import defaultdict
from itertools import product
import re
from typing import Dict, List, Tuple, Set, Any, Sequence, Union, DefaultDict

Sig = Dict[str, Set[Tuple[str,...]]]
StrT = Tuple[str,...]
# StrP = Tuple[str,str]
StrL = List[str]
StrS = Set[str]
S2S = Dict[str,str]

################################
def preds_from_yaml(filename: str) -> Sig:
    ''' Read predicates and their extensions from yaml.
        Return a dictionary where keys preserved as they are in .yaml file
    '''
    with open(filename) as F:
        d = yaml.load(F, Loader=yaml.Loader)
    sr: DefaultDict[str, Set[StrT]]  = defaultdict(set)
    for pred, args_list in d.items():
        #pred_sym, pos_arity = pred.rsplit('_', 1)
        # unary preds don't include 1, but 1 is explicitly added in signature
        if pred.endswith('_n') or pred.endswith('_a') or pred.endswith('_pn'):
        # unary predicates in a set format
            sr[pred].update( (k,) for k in args_list )
                # unary pred extension has singleton tuples for type uniformity
        elif pred == 'MOD_2':
        # Cartesian product of unary preds and its extensions
            for mods, heads in args_list:
                for m in mods:
                    sr[f"{m}_mod"].update(heads)
        else: # non-unary predicates which ends with arity
            assert isinstance(args_list, list) and re.match('.+\d$', pred),\
                f"{pred} is expected to be a non-unary predicate"
            # pred has a form verb_vN, prep_pN
            arity = int(pred[-1])
            try:
                sr[pred] = list_of_list_to_nary_rel(args_list, arity)
            except Exception as e:
                print(f"Error raised for {pred}")
                raise
    # factual/knowledge relations are automatically added to the toy world relations
    for pred, ext in sr.items():
        if pred.startswith('k_'):
            sr.get(pred[2:], set()).update(ext)
    # convert into dict to prevent accidentally adding new keys
    return dict(sr)

def domain_size_of_sr(sr):
    domain = set()
    for pred, ext in sr.items():
        if pred.endswith('_n') or pred.endswith('_pn'):
            domain.update(ext)
    return domain


def list_of_list_to_nary_rel(list_of_list, arity):
    """
    Converst a list of list into a set of tuples of size arity.
    Its functionality is inspirde by YAML and its usage of aliases and lists.
    The elements of the list of list can be:
        * a list of strings -> a tuple of strings
        * a list of lists -> recursively calls this function
        * a list of dicts -> product of dict_keys
    """
    rel = set()
    for l in list_of_list:
        if isinstance(l[0], str): # explicit arguments
            assert len(l) == arity, f"{l} violates arity {arity}"
            rel.add(tuple(l))
        elif isinstance(l[0], list): # dealing with predicate alias value
            extracted_rel = list_of_list_to_nary_rel(l, arity)
            rel.update(extracted_rel)
        elif isinstance(l[0], dict): # cartesian product of sets/dicts
            list_of_arglist = [ list(a.keys()) for a in l ]
            for new_args in product(*list_of_arglist):
                assert len(new_args) == arity, \
                    f"{new_args} violates arity {arity}"
                rel.add(tuple(new_args))
        else:
            raise RuntimeError(f"{l} has unexpected element types: it elements are not either strings, lists or dicts")
    return rel
