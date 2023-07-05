#!/usr/bin/env python3
# -*- coding: utf8 -*-

''' Printing, reproting, and writing functions
'''

import re
from itertools import zip_longest
from typing import Dict, Any, Set, Tuple, List, Sequence, OrderedDict, Iterator
from generate_utils import StrS, S2S
import json, io

SeqStr = Sequence[str]

V_LEVEL: Dict[str, int] = dict()


##############################
def framed_print(string: str, title: str, l: int = 80, sym: str = '=') -> None:
    ''' TODO
    '''
    fmt = f"{{:{sym}^{l}}}\n{{}}\n{{}}"
    print(fmt.format(' ' + title + ' ', string, sym*l))

#################################
def print_summary_replacements(tok2vals: OrderedDict[str,StrS],
    vdel: str = '-'*80) -> None:
    ''' TODO
    '''
    tok_fmt, val_list = [], []
    for tok, vals in tok2vals.items():
        m = max([ len(v) for v in vals ] + [len(tok)])
        tok_fmt.append(f"{{: ^{m}}}")
        val_list.append([tok] + sorted(vals))
    for j, values in enumerate(zip_longest(*val_list)):
        for i, v in enumerate(values):
            vv = '-' if v == '' else v
            print(tok_fmt[i].format('' if vv is None else vv), end=' ')
        print()
        if not j: print(vdel) # to delimit the template from values

#################################
def print_all_replacements(tok2vals: OrderedDict[str,StrS],
    prob_subs: Iterator[Tuple[SeqStr,S2S]]) -> None:
    ''' TODO
    '''
    tok_fmt = []
    for tok, vals in tok2vals.items():
        m = max([ len(v) for v in vals ] + [len(tok)])
        tok_fmt.append(f"{{: ^{m}}}")
    already_printed = set()
    for j, (_, subs) in enumerate(prob_subs):
        print_subs = tuple([ tok_fmt[i].format(
                             subs[t[1:-1]] if t.startswith('{') and t.endswith('}')
                             and v else '') \
                             for i, (t, v) in enumerate(tok2vals.items()) ])
        if print_subs not in already_printed:
            print(' '.join(print_subs))
            already_printed.add(print_subs)

#################################
def write_nli(object, fmt, pid, pretty_prob, sub, p):
    if fmt == 'json':
        json_write_nli(object, pid, pretty_prob, sub, p)
    elif fmt == 'txt':
        pretty_write_nli(object, pid, pretty_prob, sub, p)
    else:
        raise RuntimeError(f"Unknown file type: {fmt}")

def json_write_nli(object, pid, prob, sub_dict, pattern):
    d = dict(pattern.att)
    d['id'] = pid
    d['prem_num'] = len(prob)-1
    d["premises"] = '. '.join(prob[:-1]) + '.'
    d["hypothesis"] = prob[-1] + '.'
    d["subs"] = sub_dict
    if isinstance(object, dict):
        object["data"].append(d)
    elif isinstance(object, io.TextIOBase):
        object.write(json.dumps(d, ensure_ascii=False) + '\n')
    else:
        raise RuntimeError(f"Unknown type for writing: {type(object)}")

def pretty_write_nli(object, pid, prob, sub_dict, pattern):
    d = dict(pattern.att)
    meta = f"id:{pid}\tlabel:{d['label']}"
    premises = [ f"P{i}: {prem}" for i, prem in enumerate(prob[:-1], start=1) ]
    premises = '\n'.join(premises)
    hypothesis = f"H: {prob[-1]}"
    write = f"{meta}\n{premises}\n{hypothesis}\n\n"
    if isinstance(object, str):
        object += write
    elif isinstance(object, io.TextIOBase):
        object.write(write)
    else:
        raise RuntimeError(f"Unknown type for writing: {type(object)}")


#################################
def print_pattern_details(v, pattern, prob_subs, not_sampled_ex):
    if v >= V_LEVEL['each_problem']:
        for i, (prob, _) in enumerate(prob_subs, start=1):
            print(f"{i: >5}: {prob}")
    # print summary of or all replacements in the pattern
    if v >= V_LEVEL['summary_of_replacements']:
        print('-'*80)
        for sen_rep in pattern.tok_rep:
            if v >= V_LEVEL['summary_of_replacements']:
                print_summary_replacements(sen_rep)
            if v >= V_LEVEL['all_replacements']:
                print('/'*80)
                print_all_replacements(sen_rep, prob_subs)
            print('-'*80)
        feats = re.sub('\s*[\n;]\s*', '  ', pattern.raw_ft.strip())
        eqs = re.sub('\s*[\n;]\s*', '  ', pattern.bl.strip())
        sel_rest = re.sub('\s*[\n;]\s*', '  ', pattern.raw_sr.strip())
        print(f"Features : {feats}\nEquations: {eqs}\nSel.rest.: {sel_rest}")
    # check if the sample problem is generated
    pat_id = pattern.att['id']
    if v >= V_LEVEL['#problems_per_pattern']:
        n = len(prob_subs)
        not_sampled_msg = f" {not_sampled_ex} examples not sampled" if not_sampled_ex else ''
        print(f"Pattern {pat_id}: {n} sample problems;{not_sampled_msg}")
        print("="*80)
