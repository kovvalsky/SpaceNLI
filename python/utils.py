#!/usr/bin/env python3
# -*- coding: utf8 -*-

from collections import Counter, defaultdict
import re, random
from nltk.corpus import wordnet as wn  # type: ignore
from typing import Dict, List, Any, Sequence
import pickle
import bz2, gzip, lzma
from types import ModuleType

##############################
def unify_dicts(dict_maps: Sequence[dict], value_mode: str = 'unique') -> Dict:
    if value_mode not in ('unique', 'set', 'first', 'last'):
        raise RuntimeError(f"Unknown value_mode: {value_mode}")
    unified_mapping: Dict = dict()
    for mapping in dict_maps:
        for k, v in mapping.items():
            if k in unified_mapping:
                if value_mode == 'unique':
                    assert unified_mapping[k] == v, \
                       f"map uniquness fails: {unified_mapping[k]} != {v}"
                elif value_mode == 'set': unified_mapping[k].add(v)
                elif value_mode == 'first': pass
                elif value_mode == 'last': unified_mapping[k] = v
            else:
                if value_mode == 'set':
                    unified_mapping[k] = set([v])
                else:
                    unified_mapping[k] = v
    return unified_mapping

#############################
def split_dict_on_keys(d, keys, exclude=[]):
    """
    Return two dict where one contains keys and another rest of the keys.
    Excluded keys are ignored.
    """
    d1, d2 = dict(), dict()
    for k, v in d.items():
        if k in keys:
            if k in exclude: raise RuntimeError(f"{k} shouldn't be excluded")
            d1[k] = v
        elif k not in exclude:
            d2[k] = v
    return d1, d2


#############################
def element_ids_as_ids(elements, error=True):
    '''Checks if the element ids are really serving as ids.
       Returns a dict of duplicated IDs if any
    '''
    cnt = Counter([ e.attrib['id'] for e in elements ])
    duplicates = { k: v for (k, v) in cnt.items() if v != 1 }
    if duplicates and error:
        raise RuntimeError(f"@id are not identifiers: {duplicates}")
    return duplicates

#############################
def sorted_tuple(iter):
    return tuple(sorted(iter))

#############################
def int_prefix(s):
    m = re.match('\d+', s)
    if m:
        return int(m.group(0))

def id2int_ext(i):
    m = re.match('(\d+)(.*)$', i)
    if m:
        return int(m.group(1)), m.group(2)
    raise RuntimeError(f"{i} must have a form of /(\d+)(.*)$/")

#############################
# processing predicates from yaml signature
def type_arity_first(pred):
    sym, type_arity = pred.rsplit('_', 1)
    return type_arity, sym

def pred2arity(pred):
    if pred[-1].isdigit():
        return int(pred[-1])
    return 1

def pred2sym(pred):
    return pred.rsplit('_', 1)[0]


#############################
def random_replace(l, e, filter=None, seed=None):
    '''Replace a random element from l that satisfies the filter by e'''
    filtered = [ i for i in l if filter(i) ] if filter else l
    if not filtered:
        return l
    random_e = Random(seed).choice(filtered) if seed else random.choice(filtered)
    return [ (e if i == random_e else i) for i in l ]

#############################
def agreement_ratio(str_dist):
    total = sum([ int(i) for i in str_dist ])
    return int(str_dist[0])/total

#############################
def return_dups(l):
    s = set(l)
    if len(s) != len(l):
        return [ i for i in l if i not in s ]

################################
def wn_lemmatize(w: str) -> str:
    wn_lemma = wn.morphy(w)
    lemma = w if wn_lemma is None else wn_lemma
    return lemma

#################################
def wn_ss2lemmas(ss: str) -> List[str]:
    ''' Get all lemmas from the input synset.
        Lemmas have _ replaced with a whitespace
    '''
    assert re.match('\S+\.[snvar]\.\d+', ss), f"'{ss}' is not formatted as a synset"
    lemmas = [ l.name().replace('_', ' ') for l in wn.synset(ss).lemmas() ]
    return lemmas

#################################
def read_pickle(path, method=None):
    """ reads variable pickled in the file and returns it
    """
    if method:
        m = eval(method)
        assert isinstance(m, ModuleType),\
            f"{m} should be a module"
        F = m.open(path, 'rb')
    else:
        F = open(path, 'rb')
    v = pickle.load(F)
    F.close()
    return v

def write_pickle(variable, path, method=None):
    """ pickle a given variable in the file
    """
    if method:
        m = eval(method)
        assert isinstance(m, ModuleType),\
            f"{m} should be a module"
        F = m.open(path, "wb")
    else:
        F = open(path, 'wb')
    pickle.dump(variable, F)
    F.close()
