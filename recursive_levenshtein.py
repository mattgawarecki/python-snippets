# Recursive Levenshtein string distance
# Written after reading http://en.wikipedia.org/wiki/Levenshtein_distance and studying the example there

def recursive_levenshtein(s1, s2):
    lengths = [len(s1), len(s2)]
    if 0 in lengths:
        return max(lengths)
    else:
        cost = 0 if s1[-1] == s2[-1] else 1
        return min([recursive_levenshtein(s1[:-1], s2) + 1,
                    recursive_levenshtein(s1, s2[:-1]) + 1,
                    recursive_levenshtein(s1[:-1], s2[:-1]) + cost])
