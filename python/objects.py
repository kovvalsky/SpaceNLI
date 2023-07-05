#!/usr/bin/env python3
# -*- coding: utf8 -*-

import nltk  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from nltk.grammar import FeatureGrammar  # type: ignore
from nltk.grammar import FeatStructNonterminal as FSNT
from nltk.parse.featurechart import FeatureChart, FeatureChartParser  # type: ignore
from nltk.parse.generate import generate  # type: ignore
import re
from collections import defaultdict, OrderedDict as OrdDict
import itertools
from xml.etree.ElementTree import Element
from nltk.featstruct import FeatStruct, Feature  # type: ignore
from pattern_utils import findall_tag_text, prob2sents,\
    read_selection_restrictions
from grammar_utils import get_nt_sym
from utils import unify_dicts
from typing import Dict, List, Tuple, Sequence, Any, TypeVar, DefaultDict, Set,\
    Iterator, Optional, Union, OrderedDict, Iterable
from generate_utils import Sig, StrT, StrS, S2S
import macros
import inspect

SeqStr = Sequence[str]

# possible tokens: ${...} | $1{...} | $(...) | $(...)? | , | 's |NP1[+det] | words
# REGEXP_TOK = RegexpTokenizer("\$\d*\{[^}]+\}\??|\$\d*\([^)]+\)\??|,|'s|[A-Za-z][^,'\s]+")
REGEXP_TOK = RegexpTokenizer("{[^}]+}|,|'s|[A-Za-z][^,'\s]+")
V_LEVEL: Dict[str, int] = dict()


class VarValue:
    def __init__(self, vn, nti2fnti=dict(), nt_set={}):

        nt_regex = f"({'|'.join(nt_set)})" + r'(\d+)?$' # disable features
        ss_regex = r'(\w+)_([ansrv])_(\d+)$'
        str_regex = r"_(\w+)$"
        var_regex = r"\w+$"

        if (m := re.match(nt_regex, vn)):
            self.type = 'nt'
            self.src = m.groups()
            self.fnt = nti2fnti.get(m.group(0), FSNT(self.src[0]))
        elif (m := re.match(ss_regex, vn)):
            self.type = 'ss'
            self.src = '.'.join(m.groups())
        elif (m := re.match(str_regex, vn)):
            self.type = 'str'
            self.src = ('', m.group(1))
        elif (m := re.match(var_regex, vn)):
            self.type = 'var'
            self.src = m.group(0)
        else:
            raise RuntimeError(f"Unknown var type: {vn}")

    def __hash__(self):
        if self.type == 'nt':
            return self.fnt
        else:
            return self.src

    def __eq__(self, another):
        if self.type != another.type:
            return False
        elif self.type == 'nt':
            return self.fnt == another.fnt
        else:
            return self.src == another.src

    def __repr__(self): #TODO: make FNT representation friendly
        return str(self.__hash__())

#################################
def read_nt_features(ft_text: str, nt_set: StrS) -> Dict[str,FSNT]:
    ''' TODO
    '''
    if not nt_set: raise RuntimeError(f"No NT symbols extracted from grammar")
    # provided set of non-terminals are used to check
    # the validity of extracted feature non-terminals
    nt_regex = f"({'|'.join(nt_set)})" + r'(\d+)?$'

    # processing each featured non-terminal
    nti2fnt = {}
    type = Feature('type') # special key to get NT from FSNT
    for nti_with_ft in ft_text.replace(';', ' ').split():
        #TODO: add try here
        fnti = FSNT(nti_with_ft)
        str_nti = get_nt_sym(fnti)
        m = re.match(nt_regex, str_nti)
        if not m: raise RuntimeError(f"{str_nti} NT symbol out of grtammar's coverage")
        #nti.freeze()
        if str_nti in nti2fnt:
            raise RuntimeError(f"{str_nti} is repeated in FT")
        else:
            fnti[type] = m.group(1) # get rid of index in Featured NT
            nti2fnt[str_nti] = fnti
    return nti2fnt


##############################
def process_indexed_tokens(prob_pattern, ft_nts=[]):
    symi2ufnt = defaultdict(list) # indexed non-terminals to unified feature set
    ty_i2q_text = defaultdict(dict) # maps indices of optionals to their (type, text)
    all_tokens = [ t for s_pat in prob_pattern for t in s_pat.cfg_tokens ] + ft_nts
    for t in all_tokens:
        # an indexed expression
        # t = (q, ind, type, ext) or (non-terminal+feat, ind)
        if isinstance(t, tuple) and t[1] is not None:
            # indexed +? token or non-terminal
            i = t[1]
            if len(t) > 3 and t[3] is not None:
                # indexed +? token
                ty_i2q_text[t[2]][i] = (t[0], t[3])
            elif isinstance(t[0], FSNT):
                # and len(t[0].keys()) > 1: # indexed featured NT
                symi = (get_nt_sym(t[0]), i)
                if symi in symi2ufnt:
                    symi2ufnt[symi] = symi2ufnt[symi].unify(t[0])
                else:
                    symi2ufnt[symi] = t[0]
    return symi2ufnt, ty_i2q_text

##############################
def read_coreferring(nli_sc, ft_nts=[]):
    # NT_symbol-index -> unified feat_NT, ?+ind -> ?+, type, extent
    symi2ufnt, ty_i2q_text = process_indexed_tokens(nli_sc.pt, ft_nts=ft_nts)

    # map featured NT to the positions it needs to fill
    # [featNT, index] -> [(sen_index, tok_index), ...]
    fnt_i2pos = defaultdict(lambda: defaultdict(list))
    # index of optional to its occurence positions
    ty_i2pos = defaultdict(lambda: defaultdict(list))
    uniq_ind = -1 # assign negative ids to non-coreferring vars
    for i, s_sc in enumerate(nli_sc.pt):
        for j, t in enumerate(s_sc.cfg_tokens):
            if isinstance(t, tuple): # var token
                if isinstance(t[0], FSNT): # is a CFG-token
                    if t[1] is not None: # is indexed
                        ind = t[1]
                        fnt = symi2ufnt[(get_nt_sym(t[0]), ind)]
                    else:
                        ind = uniq_ind
                        uniq_ind -= 1
                        fnt = t[0]
                    fnt.freeze()
                    fnt_i2pos[fnt][ind].append((i, j))
                else: # is a ?+token
                    if t[1] is not None: # is indexed
                        ind = t[1]
                    else:
                        assert t[0] in (None, '?'), f"Unexpected q value: {t[0]}"
                        ind = uniq_ind
                        uniq_ind -= 1
                        ty_i2q_text[t[2]][ind] = (t[0], t[3])
                    ty_i2pos[t[2]][ind].append((i, j))
    return fnt_i2pos, ty_i2pos, ty_i2q_text


##############################
class NLI_pattern:
    ''' Pattern class for NLI problems
        nli_problem: an xml element containing p,h,pp,ph,pc elements
    '''
    def __init__(self, prob_el: Element, group_el: Optional[Element],
        sig: Sig, ignore_knw: bool = False, nt_set: StrS = set(),
        regexp_tok: RegexpTokenizer = REGEXP_TOK, v: int = 0):
        # keep a sample problem to check its existance in generated ones
        self.exs = [ prob2sents(ex, ignore_knw) for ex in findall_tag_text([prob_el], 'ex') ]
        raw_pt = findall_tag_text([prob_el], 'PT')
        if len(raw_pt) != 1:
            raise RuntimeError(f"There are {len(raw_pt)} problem patterns, expected 1")
        seg_pt = prob2sents(raw_pt[0], ignore_knw)
        self.bl = ' and '.join(findall_tag_text([prob_el, group_el], 'BL'))
        self.raw_sr = '; '.join(findall_tag_text([prob_el, group_el], 'SR'))
        # it is ok FT to have more vars than the pattern has
        self.raw_ft = '; '.join(findall_tag_text([prob_el, group_el], 'FT'))

        self.att = unify_dicts([prob_el.attrib, group_el.attrib if group_el else {} ],
                                value_mode='first')
        self.sr = read_selection_restrictions(self.raw_sr, sig, ignore_knw=ignore_knw)
        nti2fnti = read_nt_features(self.raw_ft, nt_set)

        self.pt = [ S_pattern(raw) for raw in seg_pt ]
        self.vars = set([ v for s in self.pt for v in s.vars ])
        #self.var2value = var_values(self.vars, nti2fnti, nt_set)
        self.var2value = { vn: VarValue(vn, nti2fnti, nt_set) for vn in self.vars }
        self.tok_rep: List[OrderedDict[str,StrS]] \
            = [ OrdDict( (t, set()) for t in s.toks ) for s in self.pt ]


    def substitute(self, fnt2ph_head: Dict[FSNT, S2S],
        ss2lemmas: Dict[str, SeqStr], sig: Sig, v: int = 0
        ) -> Iterator[Tuple[SeqStr, S2S]]:
        ''' Produces pairs of an instance nli problem and replacements, i.e.
            replacements are used to obtain the instance nli problem from the pattern
        '''
        # create a mapping for variables to its possible values
        # fro NTs, string values come with heads
        vars = sorted(self.vars)
        # var2values maps var names in pattern to their range of values;
        # ph2head maps phrases to their heads
        var2values: Dict[str, Union[S2S,Sequence[str]]] = {}
        ph2head: S2S = {}
        for var in vars:
            val = self.var2value[var]
            if val.type == 'nt':
                var2values[var] = fnt2ph_head[val.fnt] # dict: text->head
                ph2head = unify_dicts([ph2head, fnt2ph_head[val.fnt]], value_mode="unique")
            elif val.type == 'ss':
                var2values[var] = ss2lemmas[val.src]
            elif val.type == 'str':
                var2values[var] = val.src
            else:
                raise RuntimeError(f"{var} is of an unexpected type")

        # further restrict values for nt vars by taking into account
        # selection restrictions (from arg position projection)
        restricted_ntvar_vals(var2values, self.sr, sig)
        # variable values aligned with var names
        # possible values are sorted for determinism
        all_vals = [ sorted(var2values[v]) for v in vars ]
        if self.bl: bl = compile(self.bl, "<string>", "eval")
        #sen_pts = [ compile("f'''" + pt.text + "'''", "<string>", "eval") for pt in self.pt ]
        all_problems = set()
        # use functions defined in macros in eval and make sig available
        glob_eval_mapping = dict(inspect.getmembers(macros, inspect.isfunction))
        glob_eval_mapping = unify_dicts([glob_eval_mapping, {'sig': sig}])
        for vals in itertools.product(*all_vals):
            var_mapping = dict(zip(vars, vals))
            eval_mapping = unify_dicts([glob_eval_mapping, var_mapping])
            if self.bl:
                b = eval(bl, eval_mapping)
                # evals pushes in __builtins__
                #if '__builtins__' in var_mapping: del var_mapping['__builtins__']
                if not b: continue # ignore substitution if it fails boolean condition
            m = violates_sel_rest(self.sr, var_mapping, ph2head, sig)
            if m: continue # ignore when selection restriction is violated
            # Creating a concrete instance from a pattern
            # problem = [ eval(sen_pt, var_mapping) for sen_pt in sen_pts ]
            problem = tuple( pretty_sen(pt.text.format(**var_mapping))
                        for pt in self.pt )
            if problem in all_problems:
                raise RuntimeError(f"{problem} is generated again: {var_mapping}")
            all_problems.add(problem)
            if v >= V_LEVEL['summary_of_replacements']:
                for s in self.tok_rep:
                    for t in s:
                        if t.startswith('{') and t.endswith('}'):
                            s[t].add(var_mapping[t[1:-1]])
            yield problem, var_mapping

class S_pattern:
    '''Sentence pattern
    '''
    def __init__(self, pattern, regexp_tok=REGEXP_TOK):
        # both .toks and .cfg_tokens are tuples as immutable is safer here
        self.toks = tuple(regexp_tok.tokenize(pattern))
        self.text = pattern.strip()
        self.vars = re.findall('\{([^}]+)\}', pattern)

##############################
def pretty_sen(sen: str) -> str:
    ''' Replace multiple whitespaces with a single one,
        and uppercase the first letter.
    '''
    sen = re.sub('  +', ' ', sen)
    sen = sen[0].upper() + sen[1:]
    return sen

##############################
def violates_equations(eq, vari2txt, txt2head):
    '''r_list contains values of vars while eq expresses equations
       in terms of vars.
    '''
    equal, inside = eq
    # check equality constraints
    for (xh, yh), e  in equal.items():
        x_val = eval_ht(xh, vari2txt, txt2head)
        y_val = eval_ht(yh, vari2txt, txt2head)
        if (x_val == y_val) != e:
            return (xh, x_val), (yh, y_val)
    # check inside_a_list constraints
    for (xh, b), vals in inside.items():
        #if x in vari2txt and (vari2txt[x] in vals) != b:
        #    return x, b, vari2txt[x], vals
        x_val = eval_ht(xh, vari2txt, txt2head)
        if (x_val in vals) != b:
            return xh, b, x_val, vals

def eval_ht(xh, vari2txt, txt2head, raise_error=True):
    '''Evaluate (potentially headed) variable
    '''
    x, h = xh
    if x in vari2txt:
        return txt2head[vari2txt[x]] if h == 'h' else vari2txt[x]
    else:
        if raise_error: raise RuntimeError(f'{x} not found in {vari2txt}')


def violates_sel_rest(sr: Sig, var_mapping: S2S, ph2head: S2S, sig: Sig
    ) -> Optional[str]:
    vars = set(var_mapping)
    # str_sets = set(['n1', 'pn1'])
    for pred_arity, set_of_arglist in sr.items():
        for args in set_of_arglist:
            if not set(args).issubset(vars):
                # if selection retsriction is not fully applicable to the pattern
                # vars, then ignore it
                continue
                # raise RuntimeError((f"{args} from selection restriction "
                #                     f"is not covered by {vars}"))
            arg_heads = tuple([ ph2head[var_mapping[a]] for a in args ])
            # heads = head_args[0] if pred_arity[1] in str_sets else head_args
            # noun sets have string elements but _v and _p sets have tuple elements
            if arg_heads not in sig[pred_arity]:
                return f"{pred_arity}@{args} --> {pred_arity}@{arg_heads} violates"
    return None # to silence mypy


def sel_rest_proj(sr: Sig, sig: Sig) -> Dict[str, StrS]:
    '''Extract argument projections for existing selection restrictions, e.g.,
       from walk_accross[NP1, NP2], get permitted values of NP1 and NP2
    '''
    mapping: Dict[str, StrS] = {}
    # str_sets = set(['n1', 'pn1'])
    for pred_arity, ext in sig.items():
        # make sure that sr is not empty.
        # If it is empty, it means that there are no restrictions
        if not(pred_arity in sr and sr): continue
        projections: List[StrT] = list(zip(*ext))
            #= [ext] if pred_arity[1] in str_sets else \
            #list(zip(*ext))
        for args in sr[pred_arity]:
            assert len(args) == len(projections),\
                f"Mismatch in arg numbers in <SR> and Signature for {pred_arity}"
            for a, vals in zip(args, projections):
                if a in mapping:
                    mapping[a].intersection_update(vals)
                else:
                    mapping[a] = set(vals)
    return mapping

def restricted_ntvar_vals(var2values: Dict[str,Any], sr: Sig, sig: Sig
    ) -> None:
    """
    In-place modifies/restricts values of var2values
    taking into account selection restrictions and signature
    """
    nt_vars2heads = sel_rest_proj(sr, sig)
    for nt_var, heads in nt_vars2heads.items():
        # it is ok if SR has more vars than the pattern
        if nt_var not in var2values: continue
        # remove values whose heads don't satisfy sel restriction
        var2values[nt_var] = { txt: h for txt, h in var2values[nt_var].items()
                               if h in heads }
