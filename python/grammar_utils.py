#!/usr/bin/env python3
# -*- coding: utf8 -*-


from nltk.grammar import FeatureGrammar  # type: ignore
from nltk.grammar import FeatStructNonterminal as FSNT # type: ignore
from nltk.featstruct import Feature  # type: ignore
from nltk.parse.generate import generate  # type: ignore
# from nltk.featstruct import FeatStruct
from nltk.tree import Tree  # type: ignore
import re
from generate_utils import Sig, StrS, S2S
from typing import Dict, List, Tuple, Set, Sequence, Iterable, Any, Union
from utils import type_arity_first


#################################
def read_grammar(grammar_file: str, remove_lhs=[], sig=None, out=None)\
    -> Tuple[FeatureGrammar, StrS]:
    ''' Read grammar from a file and remove rules with certain LHS symbols.
        If a signature sig is specified, extract grammar rules from it.
        If an output file out is specified, write the grammar in the file.
    '''
    grammar = []
    with open(grammar_file) as F:
        for l in F:
            l = l.strip()
            if l.startswith('#') or not l:
                continue
            m = re.match('[A-Z]+', l)
            if m and m.group(0) in remove_lhs:
                pass
            else:
                grammar.append(l)
    if sig:
        gr_rules = sig_to_grammar_rules(sig)
        grammar += gr_rules
    if out:
        with open(out, 'w') as F:
            F.write('\n'.join(grammar))
    grammar = FeatureGrammar.fromstring('\n'.join(grammar))
    # get all non-terminal symbols from the grammar
    nt_set = grammar2nt(grammar)
    return grammar, nt_set

#################################
def sig_to_grammar_rules(sig):
    ''' Extract grammar rules for proper names, nouns and adjectives
        from dict signature and
        return a list of CFG rules with featured non-terminals
    '''
    gr_rules = []
    # proper names are grammatically different from nouns, so they need to be
    # excluded from N-rules, but first get a set of all proper names
    all_pn = set.union(*[ ext for s, ext in sig.items() if s.endswith('_pn') ])
    sorted_rel_ext = sorted(sig.items(), key=lambda x: type_arity_first(x[0]))
    for pred, ext in sorted_rel_ext:
        if pred.endswith('_n'):
            # i[0] because i is a singleton tuple
            # exclude proper names as grammatically they are different
            new_rules = [ f"N[dep='{i[0]}'] -> '{i[0]}'" for i in ext if i not in all_pn ]
            gr_rules += [ r for r in new_rules if r not in gr_rules ]
        elif pred.endswith('_a') or pred.endswith('_mod'):
            rel = pred.rsplit('_', 1)[0]
            r = f"A[dep='{rel}'] -> '{rel}'"
            if r not in gr_rules: gr_rules.append(r)
        elif pred.endswith('_pn'): # proper name
            new_rules = [ f"PN[dep='{i[0]}', +det] -> '{i[0]}'" for i in ext ]
            gr_rules += [ r for r in new_rules if r not in gr_rules ]
    return gr_rules


#################################
def generate_check_fcfg(root: FSNT, parser, sig: Sig, v=False)\
    -> Tuple[S2S, Tuple[int,int]]:
    ''' Generates all phrases of category root by the parser,
        checks if phrases are satisfied selection restrictions from adjectives,
        and return a Dict {phrase->head} with counts of generated and well-formed phrases
    '''
    gen_cnt, rec_cnt, phrase2head = 0, 0, {}
    #nt_sym = FeatStructNonterminal(nt)
    g_set = set([ tuple(g) for g in generate(parser._grammar, start=root) ])
    gen_cnt = len(g_set)
    for g in sorted(g_set):
        tree_head = fcfg_recognize_check(g, parser, root, sig)
        if tree_head:
            rec_cnt += 1
            phrase2head[' '.join(g)] = tree_head[1]
        else:
            raise RuntimeError(f"{g} and its head were not uniquely recognized as {root}")
    counts = (gen_cnt, rec_cnt)
    if v:
        rec_list = '\n'.join([ ' '.join(ph) for ph in phrase2head ])
        print(f"{root}:\n{rec_list}\n{counts}")
    return phrase2head, counts


def fcfg_recognize_check(tokens: Tuple, parser, root: FSNT, sig: Sig)\
    -> Tuple[Tree, str]:
    '''Parse a list of tokens with the parser and a root, and also check
       if it satisfies selection restriction for adjectives
       Return a tree and its head
    '''
    tree_head_list = []
    for tree in fcfg_nt_parse(parser, tokens, root=root):
        dep_tree = get_dep_tree(tree)
        if np_adj_sel_rest(dep_tree, sig):
            tree_head_list.append((tree, dep_tree.label()))
    # raise an error when phrase is syntactically ambiguous or not recognizable
    if len(tree_head_list) > 1:
        raise RuntimeError(f"{tokens} has no unique parse: {tree_head_list}")
    if len(tree_head_list) < 1:
        raise RuntimeError(f"{tokens} has no parse")
    return tree_head_list[0]


def fcfg_nt_parse(parser, input: Sequence, root: FSNT = None) -> Iterable:
    ''' Parse input sequence with a parser and root and
        return an iterable over parse trees
    '''
    # define a tree root
    if root is None: root = parser._grammar.start()
    # parse with a custom root
    chart = parser.chart_parse(input)
    return iter(chart.parses(root))


def get_head(tree):
    head = tree
    if head.height() < 3:
        return head
    for ch in tree:
        #print(f"ch = {ch}")
        nt = FeatStructNonterminal(ch.label())
        #print(f"ch nt = {nt}")
        if 'h' in nt.keys() and nt['h']:
            head = get_head(ch)
            break
    return head

def get_dep_tree(tree) -> Tree:
    ''' #TODO
    '''
    dep_tree_str = repr(tree.label()['dep'])
    pattern = '[^\s,()][^,()]*' # ('8 feet', 'tall')
    return Tree.fromstring(dep_tree_str,
                           leaf_pattern=pattern, node_pattern=pattern)

def get_dep_head(tree):
    dep_tree_str = repr(tree.label()['dep'])
    pattern = '[^\s,()][^,()]*' # ('8 feet', 'tall')
    return Tree.fromstring(dep_tree_str,
                           leaf_pattern=pattern, node_pattern=pattern)
    return dep_tree.label()

def np_adj_sel_rest(dep_tree: Tree, sig: Sig):
    '''Checks satifiability of adjective selection restriction
    '''
    if dep_tree.leaves():
        ext = sig.get(f"{dep_tree[0]}_a", None)
        if ext and dep_tree.label() not in ext:
            return False
    return True

#################################
def get_nt_sym(fnt: FSNT) -> str:
    ''' Return a main symbol of the featured non-terminal symbol
    '''
    return fnt[Feature('type')]

#################################
def grammar2nt(gr: FeatureGrammar) -> StrS:
    ''' Return a set of non-terminal symbols used in the grammar
    '''
    nt_set = set()
    for r in gr.productions():
        nt_set.add(get_nt_sym(r.lhs()))
        nt_set.update([ get_nt_sym(sym) for sym in r.rhs() if isinstance(sym, FSNT) ])
    return nt_set
