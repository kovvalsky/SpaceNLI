#!/usr/bin/env python3
# -*- coding: utf8 -*-


'''Generate Space NLI data
'''

#################################

import argparse, json, re, sys, random
#import checklist
#from checklist.editor import Editor
#from checklist.perturb import Perturb
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
#from lxml import objectify
#from lxml import etree
from os import path as op
import nltk  # type: ignore
from nltk.data import load  # type: ignore
from nltk.parse.featurechart import FeatureChart, FeatureChartParser  # type: ignore
import objects, reporting, utils
from objects import NLI_pattern
from grammar_utils import read_grammar, generate_check_fcfg
from generate_utils import preds_from_yaml, Sig, StrS, S2S
from collections import OrderedDict as OrdDict
from typing import Dict, Any, Set, Tuple, List, Sequence, OrderedDict, Optional,\
    Union
from reporting import print_summary_replacements, print_all_replacements,\
    framed_print, write_nli, print_pattern_details

SeqStr = Sequence[str]

# verbosity levels
V_LEVEL : Dict[str,int] = {
    '#problems_per_pattern': 1,
    'summary_of_replacements': 2,
    'all_replacements': 3,
    'each_problem': 4,
    'grammar': 4
}
objects.V_LEVEL = V_LEVEL
reporting.V_LEVEL = V_LEVEL

#################################
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    'nli_patterns', metavar="FILE_PATH",
        help="NLI patterns")
    # parser.add_argument(
    # '--th', required=True, metavar="FILE_PATH",
    #     help="Type hierarchy file")
    parser.add_argument(
    '--gr', required=True, metavar="FILE_PATH", default="python/base_grammar.fcfg",
        help="Grammar file")
    parser.add_argument(
    '--anno', nargs="+", metavar="LIST OF FILE_PATH",
        help="JSON files with multi-annotated problems")
    parser.add_argument(
    '--anno-agr-thr', metavar="FLOAT", type=float,
        help="Minimum ratio of the agreement among the annotators")
    parser.add_argument(
    '--sr', metavar="FILE_PATH", default='config/selection_restriction.yaml',
        help="Selection restriction file")
    parser.add_argument(
    '--no-ex', action='store_true',
        help="Don't allow generating examples")
    parser.add_argument(
    '--ids', nargs='+', metavar="LIST OF IDS",
        help="A list of pattern IDs")
    parser.add_argument(
    '--ids-regex', metavar="REGEX OF IDS", default='.+',
        help="A pythin regex string for matching IDs")
    parser.add_argument(
    '--ids-int', metavar="INT-INT", type=lambda x: tuple(map(int, x.split('-'))),
        help="All problem ids of form Num(letters+)? where Num falls in the interval")
    parser.add_argument(
    '--ignore-knw', action='store_true',
        help="Whether include or ignore knowledge explicitly as a premise")
    parser.add_argument(
    '--out', metavar="FILE_PATH",
        help="A file where generated problems will be written. File extension needs to be .txt or .json")
    parser.add_argument(
    '--out-max-seed', nargs='+', type=int, metavar="NUM [SEED]", default=[-1, 42],
        help="Max number of problems per pattern. -1 stands for all problems, 0 stands for all and only the example ones. When >0, it makes sense also to give a seed based on which the generated problems will be randomly picked")
    parser.add_argument(
    '-v', action='count', default=0,
        help="Verbosity level")

    args = parser.parse_args()

    if args.out_max_seed[0] == 0 and args.no_ex:
        raise RuntimeError(f"Inconsistent value flags:no-ex & out-max-seed")

    # get pattern IDs based on the annotations
    if args.anno and args.anno_agr_thr:
        agr_ids = agreed_problem_ids(args.anno, args.anno_agr_thr)
        args.ids = [ i for i in agr_ids if \
                     ( not args.ids or i in args.ids ) and \
                     re.match(args.ids_regex, i) ]
        if args.ids_int: # additional filter about interval ids
            args.ids = [ i for i in args.ids if \
                         utils.int_prefix(i) and \
                         args.ids_int[0] <= utils.int_prefix(i) <= args.ids_int[1] ]

    # set default seed if not provided
    if len(args.out_max_seed) == 1:
        args.out_max_seed.append(42)
    if args.out and not (args.out.endswith('.txt') or args.out.endswith('.json')):
        raise RuntimeError(f"Out path {args.out} should end with .json or .txt")
    return args


#################################
def agreed_problem_ids(json_files, thr):
    ids = []
    for f in json_files:
        with open(json_file) as F:
            anno = json.load(F)
        ids += [ i for i, val in anno.items() \
                 if utils.agreement_ratio(val["aggr"]) >= thr ]
    return ids

##############################
def generate_problems(patterns, fnt2phrase_head, ss2lemmas, sig: Sig,
                    out=None, max_seed=(-1, 42), no_ex=False, v=0):
    ''' Generate samples from the patterns. no_ex doesn't allow to generate
        predefined examples.
        max_seed[0] = 0: generate all & only examples per pattern,
                     -1: all problems, n>0: max n problems
    '''
    # replace non-terminals with texts in nli pattern
    total_nli_problems, n_sampled_exs, n_wanted_exs = 0, 0, 0
    # d will keep all data that will be dumped in a file
    if out:
        cmd = " ".join(sys.argv)
        out_fmt = out.rsplit('.', maxsplit=1)[-1]
        if out_fmt == 'json':
            d = {"generated with a command": cmd}
            d['data'] = []
        elif out_fmt == 'txt':
            d = f"generated with a command: {cmd}\n\n"
        else:
            raise RuntimeError(f"Unknown file type: {out_fmt}")
    # iteration through patterns and generating problems
    for p in patterns:
        try:
            prob2subs = OrdDict(p.substitute(fnt2phrase_head, ss2lemmas, sig, v=v))
            sel_gen_probs, sampled_exs, wanted_exs = \
                sample_from_pattern(prob2subs, p, max_seed, no_ex)
            sel_gen_probs = [ (f"{p.att['id']}-{i}", sp, prob2subs[sp], p) for
                               (i, sp) in enumerate(sel_gen_probs) ]
            if out:
                for gprob in sel_gen_probs: write_nli(d, out_fmt, *gprob)
            n_sampled_exs += len(sampled_exs)
            n_wanted_exs += len(wanted_exs)
            prob_subs = [ tuple(gp[1:3]) for gp in sel_gen_probs ]
            print_pattern_details(v, p, prob_subs, len(wanted_exs))
            total_nli_problems += len(sel_gen_probs)
        except:
            print(f"ERROR: pattern ID = {p.att['id']}")
            raise
    # Finalizing writing in file and printing stats
    if out:
        with open(out, 'w') as F:
            if out_fmt == 'json': json.dump(d, F, ensure_ascii=False, indent=2)
            if out_fmt == 'txt': F.write(d)
    return total_nli_problems, n_sampled_exs, n_wanted_exs


def sample_from_pattern(prob2subs: OrderedDict[SeqStr,S2S], p, max_seed, no_ex):
    ''' Select max_n number of samples from all possible samples satisfying
        the example-forbiding flag no_ex
        max_n<0 returns all samples, max_n=0 return all&only examples
    '''
    max_n, seed = max_seed
    # it is checked whether all example problems are generated
    # find to what extent the examples are generated
    generated_exs, wanted_exs = [], []
    for ex in p.exs:
        if ex in prob2subs:
            generated_exs.append(ex)
        else:
            wanted_exs.append(ex)
    if wanted_exs:
        f"Pattern {p.att['id']}: {len(wanted_exs)} example problems not generated"
    if max_n == 0: # interested only in all generated examples
        return generated_exs, generated_exs, wanted_exs
    # sample from all generated problems
    problems = list(prob2subs)
    if no_ex:
        for ex in generated_exs: problems.remove(ex)
    if max_n < 0:
        selected_probs = problems
    else:
        random.seed(seed)
        selected_probs = problems if len(problems) <= max_n else random.sample(problems, max_n)
        # TODO report insufficient samples per pattern
    # preppend examples in the selection list
    if not no_ex:
        selected_exs = []
        for ex in generated_exs:
            if ex in selected_probs:
                selected_probs.remove(ex)
                selected_exs.append(ex)
        selected_probs = selected_exs + selected_probs
    return selected_probs, generated_exs, wanted_exs


def problem_elements(nli_patterns_file: str) -> Dict[Element, Optional[Element]]:
    ''' Parse nli pattern xml file and
        return a Dict(problem:problem_set)
        #TODO allow groups inside groups
    '''
    # read selection restrictions from yaml file
    # read xml file
    root = ET.parse(nli_patterns_file).getroot()
    # create a dictionary of problem elements with their set elements as values
    alone_prob_pat = { p:None for p in root.findall('./problem') }
    grouped_prob_pat = { p:g for g in root.findall('./group')
                       for p in g.findall('./problem') }
    prob2grp_el: Dict[Element, Optional[Element]] \
        = {**alone_prob_pat, **grouped_prob_pat}
    # Remove patterns without PT tag
    prob2grp_el = { e: v for e, v in prob2grp_el.items() if len(e.findall("PT")) > 0 }
    #TODO check id format int+letters?
    utils.element_ids_as_ids(prob2grp_el, error=True) # sanity check of IDness
    return prob2grp_el

def problem_patterns(prob2grp_el: Dict[Element, Any], sig: Sig,
                     nt_set: StrS, ids=None, ids_regex='.+', ids_int=None,
                     ignore_knw=None, v=0):
    ''' Given Dict(problem:problem_set) obtain a list of NLI pattern objects,
        all featured non-terminals and synset:lemmas mappings
    '''
    nli_patterns = []
    feat_nt_symbols, senses = set(), set()
    # for determinism, sort problems according to the id
    for prob_el, group_el in sorted(prob2grp_el.items(),
                                    key=lambda x: utils.id2int_ext(x[0].attrib["id"])):
        # ignore a problem if its id doesn't meet constraints (if any)
        pid = prob_el.attrib['id']
        pid_int = utils.int_prefix(pid)
        if (ids and pid not in ids) or \
           (not re.match(ids_regex, pid)) or \
           (ids_int and pid_int and not(ids_int[0] <= pid_int <= ids_int[1])):
            continue # filtering
        # main part: converts xml problems into NLI pattern objects
        try:
            NLI_sc = NLI_pattern(prob_el, group_el, sig,
                            ignore_knw=ignore_knw, nt_set=nt_set, v=v)
        except:
            print(f"ERROR: Pattern ID = {pid}")
            raise
        # skip a problem if it has no shemma
        if not NLI_sc.pt: continue
        nli_patterns.append(NLI_sc)
        #feat_nt_symbols.update(NLI_sc.fnt_i2pos.keys())
        for _, value in NLI_sc.var2value.items():
            if value.type == 'nt': feat_nt_symbols.add(value.fnt)
            if value.type == 'ss': senses.add(value.src)
        #senses.update(s for q, s  in NLI_sc.ty_i2q_text['ss'].values())
    # get word senses
    ss2lemmas = { s : utils.wn_ss2lemmas(s) for s in senses }
    return nli_patterns, feat_nt_symbols, ss2lemmas

#################################
if __name__ == '__main__':
    args = parse_arguments()
    # read grammar and create parser based on it
    sig = preds_from_yaml(args.sr)
    gr, nt_set = read_grammar(args.gr, sig=sig, out=f"{args.gr}.temp")
    if args.v >= V_LEVEL['grammar']: framed_print(gr, 'grammar')
    prob2grp_el = problem_elements(args.nli_patterns)
    # collect all nli_pattern objects and all non-terminals and senses that
    # need to be generated (to make generation more efficient)
    nli_patterns, feat_nt_symbols, ss2lemmas = \
        problem_patterns(prob2grp_el, sig, nt_set, \
        ids=args.ids, ids_regex=args.ids_regex, ids_int=args.ids_int, \
        ignore_knw=args.ignore_knw, v=args.v)
    # generate phrases
    parser = FeatureChartParser(gr, trace=0, chart_class=FeatureChart)
    fnt2phrase_head = { fnt: generate_check_fcfg(fnt, parser, sig)[0] \
                        for fnt in feat_nt_symbols }
    # generating problems by replacing non-terminals with texts in patterns
    total_nli_problems, sampled_exs, wanted_exs = \
        generate_problems(nli_patterns, fnt2phrase_head, ss2lemmas, sig, \
            out=args.out, max_seed=args.out_max_seed, no_ex=args.no_ex, v=args.v)
    print(f"{total_nli_problems} sample problems from {len(nli_patterns)} patterns")
    # print about examples sampled and examples generated
    # (generated one might not end up being chosen in the final sampled problems)
    total_exs = sum([ len(s.exs) for s in nli_patterns ])
    print(f"Examples: {sampled_exs*100/total_exs:.0f}% ({sampled_exs}/{total_exs}) were sampled")
    print(f"          {(total_exs-wanted_exs)*100/total_exs:.0f}% ({total_exs-wanted_exs}/{total_exs}) were generated")
