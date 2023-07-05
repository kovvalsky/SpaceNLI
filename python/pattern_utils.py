#!/usr/bin/env python3
# -*- coding: utf8 -*-

from collections import defaultdict
import re
from grammar_utils import get_nt_sym
from generate_utils import Sig, StrT, StrL
import itertools
from utils import sorted_tuple, pred2arity, pred2sym
from xml.etree.ElementTree import Element
from typing import Dict, List, Tuple, Set, Sequence, Iterable, Any,\
    Optional, Union, DefaultDict


################################
def findall_tag_text(parents: List[Optional[Element]], tag: str) -> StrL:
    ''' Collects contents of tag found in parents elements.
        It is used to combine tags of a problem element and its group element
    '''
    found = [ e.text.strip() for p in parents if p is not None
             for e in p.findall(tag) if e.text is not None ]
    return found

def prob2sents(problem_example: str, ignore_knw: bool) -> StrT:
    ''' Parse example problem string into tuple of sentences
    '''
    sents = [ s.strip().lstrip('^') for s in problem_example.strip().split('\n') \
                        if not(ignore_knw and s.strip()[0] == '^') ]
    return tuple(sents)

# def ht(s):
#     '''Make a headed tuple'''
#     if s.endswith('.h'):
#         return (s[:-2], 'h')
#     return (s, None)

##############################
# def read_equations(text):
#     equal = dict() #  { ((a,Head), (b,Head)) -> True/False }
#     inside = defaultdict(list) # { ((a,Head), True/False) -> list }
#     for exp in text.split(';'):
#         # $2 != $1
#         m1 = re.search('([\w$\.h]+)\s*(.?=)\s*([\w$\.h]+)', exp)
#         # !=(NP1, $1, NP3)
#         m2 = re.search('(.?=)\s*\(([\[\w$\.h\], ]+)\)', exp) #VERIFY
#         # $1 not in [now, at_once]
#         m3 = re.search('([\w$\.h]+)\s+(not\s+)?(in)\s+\[([^]]+)\]', exp)
#         if m1:
#             a, e, b = m1.groups()
#             equal[sorted_tuple([ht(a), ht(b)])] = (e == '=')
#         elif m2:
#             e, args = m2.groups()
#             args = sorted(re.split('\s*,\s*', args))
#             for i, a in enumerate(args):
#                 for b in args[i+1:]:
#                     if (ht(a), ht(b)) in equal:
#                         assert equal[(ht(a), ht(b))] == (e == '='), \
#                             f'Conflict in constrainst with {(a, e, b)}'
#                     else:
#                         equal[(ht(a), ht(b))] = (e == '=')
#         elif m3:
#             a, nnot, _in, l = m3.groups()
#             key = (ht(a), not bool(nnot))
#             for e in re.split('\s*,\s*', l):
#                 inside[key].append(e.replace('_', ' '))
#     return equal, inside

##############################
# def add_nt_inequality(eq_dict, list_fnt_i):
#     ''' By default all different NT vars are considered as inequal
#         modulo heads(!), e.g., NP1="the dog" & NP2="the black dog" is not allowed
#     '''
#     list_nti = [ f"{get_nt_sym(fnt)}{i}" for fnt, i in list_fnt_i ]
#     for nti1, nti2 in itertools.product(list_nti, list_nti):
#         if nti1 == nti2: continue
#         if (nti1, nti2) in eq_dict: continue
#         eq_dict[sorted_tuple(((nti1,'h'), (nti2,'h')))] = False


##############################
def read_selection_restrictions(text: str, sig: Sig, ignore_knw: bool = False
    ) -> Sig:
    ''' Parses selection restriction string and validates its predicates
        wrt a signature sig. Returns a dictionary pred->list of arg tuples
    '''
    sel_rest: List[Tuple[str,StrL]] = []
    for exp in re.split('\s*;\s*', text.strip()):
        if not exp: continue
        m = re.search('(\w+)\(([^)]+)\)', exp)
        if m:
            pred, args = m.groups()
            arg_list: StrL = []
            for a in re.split('\s*,\s*', args):
                # TODO: distinguish head from literal
                # nti, l = re.match('([A-Z]+[0-9]+)\.?(\w+)?', a.strip()).groups()
                m_arg = re.match('[A-Z]+[0-9]+$', a)
                if m_arg is None: raise RuntimeError(f"{a} is ill-formed arg")
                arg_list.append(a)
            sel_rest.append((pred, arg_list))
        else:
            raise RuntimeError(f"{exp} in <SR> was not parsed")
    # get arity for verbs
    sr: DefaultDict[str, Set[StrT]] = defaultdict(set)
    for pred, args in sel_rest:
        # ignore knowledge related constraints if knowledge premises are ignored
        if ignore_knw and pred.startswith('k_'): continue
        n = str(len(args))
        sym_tys = [ sym_ty for sym_ty in sig if \
                pred2sym(sym_ty) == pred and pred2arity(sym_ty) == len(args) ]
        if len(sym_tys) == 0:
            raise RuntimeError(f"Predicate '{pred}/{n}' is not found in the signature")
        elif len(sym_tys) > 1:
            raise RuntimeError(f"Predicate '{pred}/{n}' is ambiguous: {sym_tys}")
        else:
            sr[sym_tys[0]].add(tuple(args))
    return { k: v for k, v in sr.items() }
