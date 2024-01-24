# A dictionary of movie critics and their ratings of a small#
critics = {
    'A': {'老炮儿':3.5,'唐人街探案': 1.0},
    'B': {'老炮儿':2.5,'唐人街探案': 3.5,'星球大战': 3.0, '寻龙诀': 3.5,
                  '神探夏洛克': 2.5, '小门神': 3.0},
    'C': {'老炮儿':3.0,'唐人街探案': 3.5,'星球大战': 1.5, '寻龙诀': 5.0,
                     '神探夏洛克': 3.0, '小门神': 3.5},
    'D': {'老炮儿':2.5,'唐人街探案': 3.5,'寻龙诀': 3.5, '神探夏洛克': 4.0},
    'E': {'老炮儿':3.5,'唐人街探案': 2.0,'星球大战': 4.5, '神探夏洛克': 3.5,
                     '小门神': 2.0},
    'F': {'老炮儿':3.0,'唐人街探案': 4.0,'星球大战': 2.0, '寻龙诀': 3.0,
                     '神探夏洛克': 3.0, '小门神': 2.0},
    'G': {'老炮儿':4.5,'唐人街探案': 1.5,'星球大战': 3.0, '寻龙诀': 5.0,
                      '神探夏洛克': 3.5}
    }

print(critics['B']['星球大战'])
from math import sqrt

# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs, person1, person2):
    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]: si[item] = 1
    # if they have no ratings in common, return 0
    if len(si) == 0: return 0
    # Add up the squares of all the differences
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2)
                          for item in prefs[person1] if item in prefs[person2]])
    return 1 / (1 + sqrt(sum_of_squares))

print(sim_distance(critics, 'A', 'B'))

# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs, person, similarity=sim_distance):
    totals = {}
    simSums = {}
    for other in prefs:
        # don't compare me to myself
        if other == person: continue
        sim = similarity(prefs, person, other)
        # ignore scores of zero or lower
        if sim <= 0: continue
        for item in prefs[other]:
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item] == 0:
                # Similarity * Score
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # Sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim
    # Create the normalized list
    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings
print(getRecommendations(critics, 'A'))