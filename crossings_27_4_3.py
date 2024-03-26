import numpy as np

def fl(l):
    ''' Unpack a nested list of depth 2 into a flat, non-nested list.'''

    return [element for sublist in l for element in sublist]

def cr(crossings_obj):
    ''' Obtain the total number of crossings from a crossing object.'''

    crossings = 0
    for check in crossings_obj[1]:
        crossings += check[2]
    return(crossings)


# checks of a 3 repetition code (classical linear code):

rep3_checks = [ (1, 0), (1, 2),
                (3, 2), (3, 4)]

# Below are the 3 different embeddings of the [[27, 4, 3]] HGP code

#------------------------------------------------------------------------------
# Crossing layout for embedding A1
#------------------------------------------------------------------------------

hamming_A1 = [(2, 0), (2, 1), (2, 3), (2, 4),
              (5, 1), (5, 4), (5, 6), (5, 7),
              (8, 3), (8, 4), (8, 7), (8, 9)]

# crossings for long-range (non-neighboring) gates:
lr_1 = [[(i, 2), (i, 0), 1] for i in [0, 2]] + [[(i, 0), (i, 2), 1] for i in [1, 3]] 
lr_2 = [[(i, 2), (i, 4), 1] for i in [2, 4]] + [[(i, 4), (i, 2), 1] for i in [1, 3]] 
lr_3 = [[(i, 5), (i, 1), 3] for i in [2, 4]] + [[(i, 1), (i, 5), 3] for i in [1, 3]] 
lr_4 = [[(i, 5), (i, 7), 1] for i in [2, 4]] + [[(i, 7), (i, 5), 1] for i in [1, 3]] 
lr_5 = [[(i, 8), (i, 4), 3] for i in [0, 2]] + [[(i, 4), (i, 8), 3] for i in [1, 3]] 
lr_6 = [[(i, 8), (i, 3), 4] for i in [0, 2]] + [[(i, 3), (i, 8), 4] for i in [1, 3]] 


# crossings for short-range (neighboring) gates:
sr_1 = [[[(i, 1), (i - 1, 1), 1], [(i, 1), (i + 1, 1), 1]] for i in [1, 3]]
sr_2 = [[[(i - 1, 2), (i, 2), 1], [(i + 1, 2), (i, 2), 1]] for i in [1, 3]]
sr_3 = [[[(i, 3), (i - 1, 3), 2], [(i, 3), (i + 1, 3), 2]] for i in [1, 3]]
sr_4 = [[[(i, 4), (i - 1, 4), 2], [(i, 4), (i + 1, 4), 2]] for i in [1, 3]]
sr_5 = [[[(i - 1, 5), (i, 5), 2], [(i + 1, 5), (i, 5), 2]] for i in [1, 3]]
sr_6 = [[[(i, 6), (i - 1, 6), 3], [(i, 6), (i + 1, 6), 3]] for i in [1, 3]]
sr_7 = [[[(i, 7), (i - 1, 7), 2], [(i, 7), (i + 1, 7), 2]] for i in [1, 3]]

# merge crossings
embedding_A1 = [
    'A1',
    lr_1 + lr_2 + lr_3 + lr_4 + lr_5 + lr_6\
    + fl([*map(fl, [sr_1, sr_2, sr_3, sr_4, sr_5, sr_6, sr_7])])
    ]

A1_zlog_obs = [6, 9, 7, 0]

A1_xlog_obs = [[(2, i) for i in [0, 1, 6]],
               [(2, i) for i in [0, 3, 9]],
               [(2, i) for i in [6, 7, 9]],
               [(2, i) for i in [6, 9, 0, 4]]]

A1_log_obs = {'x': A1_xlog_obs, 'z' : A1_zlog_obs}
cr_A1 = cr(embedding_A1)


def crossing_mapper(crossing_object, map):
    """Maps the crossings of an embedding to the data and check qubit layout of
    of the A1 embedding. This is necessary in order to keep the overall gate order of the 
    code the same, only change the position of the errors."""

    og_map = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    remaped_corssings = [crossing_object[0], []]
    for crossing in crossing_object[1]:
        remaped_corssings[1].append([(crossing[0][0], og_map[np.where(map == crossing[0][1])[0][0]]),
                                     (crossing[1][0], (og_map[np.where(map == crossing[1][1])[0][0]]) ), crossing[2]])

    return remaped_corssings



#------------------------------------------------------------------------------
# Crossing layout for embedding A0
#------------------------------------------------------------------------------

hamming_A0 = [(1, 0), (1, 7), (1, 8), (1, 9),
              (2, 0), (2, 5), (2, 6), (2, 7),
              (3, 0), (3, 4), (3, 5), (3, 9)]

map_A0 = np.array([8, 9, 1, 7, 0, 3, 4, 5, 2, 6])

# crossings for long-range (non-neighboring) gates:
lr_1 = [[(i, 1), (i, 7), 5] for i in [0, 2]] + [[(i, 7), (i, 1), 5] for i in [1, 3]] 
lr_2 = [[(i, 1), (i, 8), 6] for i in [0, 2]] + [[(i, 8), (i, 1), 6] for i in [1, 3]] 
lr_3 = [[(i, 1), (i, 9), 7] for i in [0, 2]] + [[(i, 9), (i, 1), 7] for i in [1, 3]] 
lr_4 = [[(i, 2), (i, 5), 2] for i in [0, 2]] + [[(i, 5), (i, 2), 2] for i in [1, 3]] 
lr_5 = [[(i, 2), (i, 6), 3] for i in [0, 2]] + [[(i, 6), (i, 2), 3] for i in [1, 3]] 
lr_6 = [[(i, 2), (i, 7), 4] for i in [0, 2]] + [[(i, 7), (i, 2), 4] for i in [1, 3]] 
lr_7 = [[(i, 2), (i, 0), 1] for i in [2, 4]] + [[(i, 0), (i, 2), 1] for i in [1, 3]] 
lr_8 = [[(i, 3), (i, 0), 2] for i in [2, 4]] + [[(i, 0), (i, 3), 2] for i in [1, 3]] 
lr_9 = [[(i, 3), (i, 5), 1] for i in [2, 4]] + [[(i, 5), (i, 3), 1] for i in [1, 3]] 
lr_10= [[(i, 3), (i, 9), 5] for i in [2, 4]] + [[(i, 9), (i, 3), 5] for i in [1, 3]] 

# crossings for short-range (neighboring) gates:
sr_1 = [[[(i - 1, 1), (i, 1), 2], [(i + 1, 1), (i, 1), 2]] for i in [1, 3]]
sr_2 = [[[(i - 1, 2), (i, 2), 4], [(i + 1, 2), (i, 2), 4]] for i in [1, 3]]
sr_3 = [[[(i - 1, 3), (i, 3), 6], [(i + 1, 3), (i, 3), 6]] for i in [1, 3]]
sr_4 = [[[(i, 4), (i - 1, 4), 8], [(i, 4), (i + 1, 4), 8]] for i in [1, 3]]
sr_5 = [[[(i, 5), (i - 1, 5), 6], [(i, 5), (i + 1, 5), 6]] for i in [1, 3]]
sr_6 = [[[(i, 6), (i - 1, 6), 5], [(i, 6), (i + 1, 6), 5]] for i in [1, 3]]
sr_7 = [[[(i, 7), (i - 1, 7), 3], [(i, 7), (i + 1, 7), 3]] for i in [1, 3]]
sr_8 = [[[(i, 8), (i - 1, 8), 2], [(i, 8), (i + 1, 8), 2]] for i in [1, 3]]

# merge crossings
embedding_A0 = [
    'A0',
    lr_1 + lr_2 + lr_3 + lr_4 + lr_5 + lr_6 +lr_7 + lr_8 + lr_9 + lr_10\
    + fl([*map(fl, [sr_1, sr_2, sr_3, sr_4, sr_5, sr_6, sr_7 + sr_8])])
    ]

# map them to the A1 embedding's data and check qubit layout
mapped_embedding_A0 = crossing_mapper(embedding_A0, map_A0)

A0_zlog_obs = [4, 6, 5, 8]

A0_xlog_obs = [[(2, i) for i in [8, 9, 4]],
               [(2, i) for i in [8, 7, 6]],
               [(2, i) for i in [4, 5, 6]],
               [(2, i) for i in [4, 6, 8, 0]]]

A0_log_obs = {'x': A0_xlog_obs, 'z' : A0_zlog_obs}

cr_A0 = cr(embedding_A0)

#------------------------------------------------------------------------------
# Crossing layout for embedding A2
#------------------------------------------------------------------------------

hamming_A2 = [(2, 0), (2, 1), (2, 5), (2, 6),
              (4, 1), (4, 3), (4, 5), (4, 8),
              (7, 5), (7, 6), (7, 8), (7, 9)]

map_A2 = np.array([0, 6, 2, 1, 5, 7, 9,  8, 4, 3])

# crossings for long-range (non-neighboring) gates:
lr_1 = [[(i, 2), (i, 0), 1] for i in [2, 4]] + [[(i, 0), (i, 2), 1] for i in [1, 3]] 
lr_2 = [[(i, 2), (i, 5), 2] for i in [2, 4]] + [[(i, 5), (i, 2), 2] for i in [1, 3]] 
lr_3 = [[(i, 2), (i, 6), 3] for i in [2, 4]] + [[(i, 6), (i, 2), 3] for i in [1, 3]] 
lr_4 = [[(i, 7), (i, 9), 1] for i in [2, 4]] + [[(i, 9), (i, 7), 1] for i in [1, 3]] 
lr_5 = [[(i, 4), (i, 1), 2] for i in [0, 2]] + [[(i, 1), (i, 4), 2] for i in [1, 3]] 
lr_6 = [[(i, 4), (i, 8), 3] for i in [0, 2]] + [[(i, 8), (i, 4), 3] for i in [1, 3]] 
lr_7 = [[(i, 7), (i, 5), 1] for i in [0, 2]] + [[(i, 5), (i, 7), 1] for i in [1, 3]] 

# crossings for short-range (neighboring) gates:
sr_1 = [[[(i, 1), (i - 1, 1), 1], [(i, 1), (i + 1, 1), 1]] for i in [1, 3]]
sr_2 = [[[(i - 1, 2), (i, 2), 1], [(i + 1, 2), (i, 2), 1]] for i in [1, 3]]
sr_3 = [[[(i, 3), (i - 1, 3), 3], [(i, 3), (i + 1, 3), 3]] for i in [1, 3]]
sr_4 = [[[(i - 1, 4), (i, 4), 2], [(i + 1, 4), (i, 4), 2]] for i in [1, 3]]
sr_5 = [[[(i, 5), (i - 1, 5), 2], [(i, 5), (i + 1, 5), 2]] for i in [1, 3]]
sr_6 = [[[(i, 6), (i - 1, 6), 2], [(i, 6), (i + 1, 6), 2]] for i in [1, 3]]
sr_7 = [[[(i - 1, 7), (i, 7), 1], [(i + 1, 7), (i, 7), 1]] for i in [1, 3]]
sr_8 = [[[(i, 8), (i - 1, 8), 1], [(i, 8), (i + 1, 8), 1]] for i in [1, 3]]

# merge crossings
embedding_A2 = [
    'A2',
    lr_1 + lr_2 + lr_3 + lr_4 + lr_5 + lr_6 + lr_7\
    + fl([*map(fl, [sr_1, sr_2, sr_3, sr_4, sr_5, sr_6, sr_7, sr_8])])
    ]

# map them to the A1 embedding's data and check qubit layout
mapped_embedding_A2 = crossing_mapper(embedding_A2, map_A2)

A2_zlog_obs = [9, 3, 8, 0]

A2_xlog_obs = [[(2, i) for i in [0, 6, 9]],
               [(2, i) for i in [0, 1, 3]],
               [(2, i) for i in [9, 8, 3]],
               [(2, i) for i in [9, 3, 0, 5]]]

A2_log_obs = {'x': A2_xlog_obs, 'z' : A2_zlog_obs}

cr_A2 = cr(embedding_A2)

#------------------------------------------------------------------------------
# Crossing layout for embedding A3
#------------------------------------------------------------------------------

hamming_A3 = [(1, 0), (1, 2), (1, 3), (1, 4),
              (5, 4), (5, 6), (5, 3), (5, 7),
              (8, 7), (8, 9), (8, 3), (8, 2)]

map_A3= np.array([9, 7, 8, 2, 3, 5, 6, 4, 1, 0])

# crossings for long-range (non-neighboring) gates:
lr_1 = [[(i, 1), (i, 3), 1] for i in [0, 2]] + [[(i, 3), (i, 1), 1] for i in [1, 3]] 
lr_2 = [[(i, 1), (i, 4), 2] for i in [0, 2]] + [[(i, 4), (i, 1), 2] for i in [1, 3]] 
lr_3 = [[(i, 5), (i, 3), 1] for i in [2, 4]] + [[(i, 3), (i, 5), 1] for i in [1, 3]] 
lr_4 = [[(i, 5), (i, 7), 1] for i in [0, 2]] + [[(i, 7), (i, 5), 1] for i in [1, 3]] 
lr_5 = [[(i, 8), (i, 3), 4] for i in [2, 4]] + [[(i, 3), (i, 8), 4] for i in [1, 3]] 
lr_6 = [[(i, 8), (i, 2), 5] for i in [2, 4]] + [[(i, 2), (i, 8), 5] for i in [1, 3]] 

# crossings for short-range (neighboring) gates:
sr_1 = [[[(i, 2), (i - 1, 2), 2], [(i, 2), (i + 1, 2), 2]] for i in [1, 3]]
sr_2 = [[[(i, 3), (i - 1, 3), 2], [(i, 3), (i + 1, 3), 2]] for i in [1, 3]]
sr_3 = [[[(i, 4), (i - 1, 4), 3], [(i, 4), (i + 1, 4), 3]] for i in [1, 3]]
sr_4 = [[[(i - 1, 5), (i, 5), 2], [(i + 1, 5), (i, 5), 2]] for i in [1, 3]]
sr_5 = [[[(i, 6), (i - 1, 6), 3], [(i, 6), (i + 1, 6), 3]] for i in [1, 3]]
sr_6 = [[[(i, 7), (i - 1, 7), 2], [(i, 7), (i + 1, 7), 2]] for i in [1, 3]]

# merge crossings
embedding_A3 = [
    'A3',
    lr_1 + lr_2 + lr_3 + lr_4 + lr_5 + lr_6\
    + fl([*map(fl, [sr_1, sr_2, sr_3, sr_4, sr_5, sr_6])])
    ]

# map them to the A1 embedding's data and check qubit layout
mapped_embedding_A3 = crossing_mapper(embedding_A3, map_A3)

A3_zlog_obs = [0, 6, 4 ,9]

A3_xlog_obs = [
               [(2, i) for i in [9, 2, 0]],
               [(2, i) for i in [9, 7, 6]],
               [(2, i) for i in [0, 4, 6]],
               [(2, i) for i in [0, 6, 9, 3]]
               ]

A3_log_obs = {'x': A3_xlog_obs, 'z' : A3_zlog_obs}

cr_A3 = cr(embedding_A3)