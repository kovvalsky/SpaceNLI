#!/usr/bin/env python3
# -*- coding: utf8 -*-

#################################
# used in BL tags
def list_is_subset(l, s):
    return set(l) <= s


def diff_values(list):
    """
    True iff all values in list are different
    """
    if len(list) == len(set(list)):
        return True
