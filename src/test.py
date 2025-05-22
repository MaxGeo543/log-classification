from typing import List
import itertools


def longest_common_substring(s1: str, s2: str):
    n = len(s1)
    m = len(s2)

    dp = [[0]*(m+1) for _ in range(n+1)]
    longest = 0
    end_pos = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0
    
    return s1[end_pos - longest:end_pos]

def strsim(s1: str, s2: str):
    l1, l2 = [], []
    common = []
    lcs = longest_common_substring(s1, s2)
    if lcs == "":
        return [s1] if s1 != "" else [], [s2] if s2 != "" else [], []

    spl1 = s1.split(lcs, 1)
    spl2 = s2.split(lcs, 1)
    
    rec1 = strsim(spl1[0], spl2[0])
    l1.extend(rec1[0])
    l2.extend(rec1[1])
    common.extend(rec1[2])

    common.append(lcs)
    l1.append("")
    l2.append("")

    rec2 = strsim(spl1[1], spl2[1])
    l1.extend(rec2[0])
    l2.extend(rec2[1])
    common.extend(rec2[2])

    return l1, l2, common



def longest_common_substring_multi(strings: List[str]) -> str:
    if not strings:
        return ""
    
    base = min(strings, key=len)
    n = len(base)
    longest = ""

    for i in range(n):
        for j in range(i + 1, n + 1):
            candidate = base[i:j]
            if len(candidate) <= len(longest):
                continue
            if all(candidate in s for s in strings):
                longest = candidate

    return longest


def strsim_multi(strings: List[str]):
    if all(s == "" for s in strings):
        return [[] for _ in strings], []

    lcs = longest_common_substring_multi(strings)
    if lcs == "":
        return [[s] if s else [] for s in strings], []

    # Split all strings on the first occurrence of lcs
    splits = [s.split(lcs, 1) for s in strings]
    left_parts = [s[0] for s in splits]
    right_parts = [s[1] for s in splits]

    # Recurse on left and right parts
    unmatched_left, common_left = strsim_multi(left_parts)
    unmatched_right, common_right = strsim_multi(right_parts)

    # Combine unmatched and common parts
    unmatched = [ul + [""] + ur for ul, ur in zip(unmatched_left, unmatched_right)]
    common = common_left + [lcs] + common_right

    return unmatched, common













def str_comp_jaccard(s1: str, s2: str) -> float:
    r1, r2, common = strsim(s1, s2)
    r1 = [s for s in r1 if s != ""]
    r2 = [s for s in r2 if s != ""]
    return len(common) / (len(r1)+len(r2)+len(common))

def str_comp_dice(s1: str, s2: str) -> float:
    r1, r2, common = strsim(s1, s2)
    r1 = [s for s in r1 if s != ""]
    r2 = [s for s in r2 if s != ""]
    return 2*len(common) / (len(r1)+len(r2)+2*len(common))



def str_comp_jaccard_multi(*s) -> float:
    unmatched, common = strsim_multi(*s)
    unmatched = [len([part for part in ums if part != ""]) for ums in unmatched]
    return len(common) / (sum(unmatched) + len(common))

def str_comp_dice_multi(*s) -> float:
    unmatched, common = strsim_multi(*s)

    unmatched = [len([part for part in ums if part != ""]) for ums in unmatched]
    return len(s)*len(common) / (sum(unmatched) + len(s)*len(common))











s1 = "HliSession::Dispose(): 10.239.134.85:40001 - 'HS200_03028' -> True"
s2 = "HliSession::Close(): 10.239.134.85:40001 - 'HS200_03028'"
s3 = "HliSession::CloseInternal(): 10.239.134.85:40001 - 'HS200_03028' -> "

s = [s1, s2, s3]

multi_score = (str_comp_jaccard_multi(s), str_comp_dice_multi(s))
print(f"Multi score for all three: {multi_score}")
print()
for pair in itertools.combinations(s, 2):
    print(f"score for {pair}")
    score = (str_comp_jaccard(*pair), str_comp_dice(*pair))
    print(score)
    print()









quit()


s1 = "The cat walked"
s2 = "The cat jumpered"
r1, r2, common = strsim(s1, s2)

print(r1)
print(r2)
print(common)

print(len([s for s in r1 if s != ""]), len([s for s in r2 if s != ""]), len(common))
print(str_comp_jaccard(s1, s2), str_comp_dice(s1, s2), str_cmp(s1, s2))