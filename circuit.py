import numpy as np
import stim 

#------------------------------------------------------------------------------
# Gate operations bundled with errors
#------------------------------------------------------------------------------

def append_anti_basis_error(
                circuit : stim.Circuit,
                targets : list[int],
                p : float,
                basis: str) -> None:
    
    if p > 0:
        if basis == "X":
            circuit.append("Z_ERROR", targets, p)
        else:
            circuit.append("X_ERROR", targets, p)


def append_reset(
            circuit : stim.Circuit,
            targets : list[int],
            after_reset_flip_probability : float,
            basis: str = 'Z'
        ) -> None:

    circuit.append_operation('R' + basis, targets)
    append_anti_basis_error(circuit, targets, after_reset_flip_probability, basis)

def append_gate_1(
                circuit : stim.Circuit,
                name : str,
                targets : list[int],
                after_clifford_depolarization : float
            ) -> None:
        
    circuit.append(name, targets)
    if after_clifford_depolarization > 0:
        # we multiply by 3/4 so maximal mixing happens at p = 1
        circuit.append('DEPOLARIZE1', targets, after_clifford_depolarization * 3/4)
        

def append_gate_2(
                circuit : stim.Circuit,
                name : str,
                targets : list[int],
                after_clifford_depolarization : float, 
                Cr : int
            ) -> None:
        
        circuit.append(name, targets)
        if after_clifford_depolarization > 0:
            # account for the number of crossings:
            for _ in range(Cr):
                # we multiply by 15/16 so maximal mixing happens at p = 1
                circuit.append('DEPOLARIZE2', targets, after_clifford_depolarization*15/16)

def append_MR(
            circuit : stim.Circuit,
            targets : list[int],
            before_measure_flip_probability : float,
            after_reset_flip_probability : float,
            basis : str = 'Z'
            ) -> None:
        
        # first, measurement error:
        append_anti_basis_error(circuit, targets, before_measure_flip_probability, basis)
        
        circuit.append('MR' + basis, targets)

        # second, reset error:
        append_anti_basis_error(circuit, targets, after_reset_flip_probability, basis)


def append_M(
          circuit : stim.Circuit,
          targets : list[int],
          before_measure_flip_probability : float,
          basis : str = 'Z'
) -> None:
    
    append_anti_basis_error(circuit, targets, before_measure_flip_probability, basis)
    circuit.append('M' + basis, targets)


#------------------------------------------------------------------------------
# circuit building
#------------------------------------------------------------------------------


def circuit_builder(
        shape,
        row_checks,
        col_checks,
        crossings,
        rounds,
        experiment = 'z_memory',
        observable = False,
        after_crossing_depolarization : float = 0,
        after_clifford_depolarization : float = 0,
        after_reset_flip_probability : float = 0,
        before_measure_flip_probability : float = 0,
):
    
    # data preparation: 
    #--------------------------------------------------------------------------
    
    row = np.zeros(shape[1], dtype = int)
    col = np.zeros(shape[0], dtype = int)

    for check in row_checks:
        row[check[0]] = 1
    for check in col_checks:
        col[check[0]] = 1

    # initialize matrix that describes the positions of the data and check qubits
    qubit_matrix = np.empty(shape = shape, dtype = str)

    # dictionaries for coordinates and qubit indexes:
    data_qubits = {} 
    x_checks = {}
    z_checks = {}
    counter  = 0

    for i, col_val in enumerate(col): 
        for j, row_val in enumerate(row):
            
            if row_val == col_val:
                qubit_matrix[i][j] = 'Q'
                data_qubits.update({(i, j) : counter})
                counter += 1


            if row_val == 0 and col_val == 1 :
                qubit_matrix[i][j] = 'X'
                x_checks.update({(i, j) : counter})
                counter += 1

            if row_val == 1 and col_val == 0 :
                qubit_matrix[i][j] = 'Z'
                z_checks.update({(i, j) : counter })
                counter += 1

    all_coords  = data_qubits  | x_checks | z_checks

    basis = 'X' if experiment == 'x_memory' else 'Z'

    # circuit assembly
    #--------------------------------------------------------------------------

    # cycle to repeat:
    def cycle(
              after_clifford_depolarization = after_clifford_depolarization,
              after_crossing_depolarization = after_crossing_depolarization,
              before_measure_flip_probability = before_measure_flip_probability,
              after_reset_flip_probability = after_reset_flip_probability
              ):
        circ_cycle = stim.Circuit()
        
        append_gate_1(circ_cycle, 'H', [*x_checks.values()],  after_clifford_depolarization)

        global z_pairings
        z_pairings = {}
        global x_pairings
        x_pairings = {}
        
        all_gates = []
        z_gates = []
        x_gates = []

        # iterate through the qubit matrix:
        for i in range(qubit_matrix.shape[0]):
            for j in range(qubit_matrix.shape[1]):

                
                for c_c in col_checks:
                    if c_c[0] == i:

                        if qubit_matrix[i][j] == 'Q':
                            error = after_clifford_depolarization
                            Cr = 1
                            if crossings:
                                for crossing in crossings:
                                    if crossing[0] == (c_c[1], j) and crossing[1] == (i, j):
                                        error = after_crossing_depolarization
                                        Cr = crossing[2]
                                    
                                all_gates.append([[all_coords[(i, j)], all_coords[(c_c[1], j)]], error, Cr])
                                z_gates.append([[all_coords[(i, j)], all_coords[(c_c[1], j)]], error, Cr])
 
                            else:           
                                all_gates.append([[all_coords[(i, j)], all_coords[(c_c[1], j)]], after_clifford_depolarization, 1])
                                z_gates.append([[all_coords[(i, j)], all_coords[(c_c[1], j)]], after_clifford_depolarization, 1])

                            if (c_c[1], j) in z_pairings:
                                z_pairings[(c_c[1], j)].append((i, j))
                            else:
                                z_pairings[(c_c[1], j)] = [(i, j)]


                        else:
                            error = after_clifford_depolarization
                            Cr = 1
                            if crossings:
                                for crossing in crossings:
                                    if crossing[1] == (c_c[1], j) and crossing[0] == (i, j):
                                        error = after_crossing_depolarization
                                        Cr = crossing[2]
                                
                                all_gates.append([[all_coords[(i, j)], all_coords[(c_c[1], j)]], error, Cr])
                                x_gates.append([[all_coords[(i, j)], all_coords[(c_c[1], j)]], error, Cr])
                            
                            else:           
                                all_gates.append([[all_coords[(i, j)], all_coords[(c_c[1], j)]], after_clifford_depolarization, 1])
                                x_gates.append([[all_coords[(i, j)], all_coords[(c_c[1], j)]], after_clifford_depolarization, 1])

                            if (i, j) in x_pairings:
                                x_pairings[(i, j)].append((c_c[1], j))
                            else:
                                x_pairings[(i, j)] = [(c_c[1], j)]

                
                for r_c in row_checks:
                    if r_c[0] == j:
                        if qubit_matrix[i][j] == 'Q':
                            error = after_clifford_depolarization
                            Cr = 1
                            if crossings:
                                for crossing in crossings:
                                    if crossing[0] == (i, r_c[1]) and crossing[1] == (i, j):
                                        error = after_crossing_depolarization
                                        Cr = crossing[2]
                                        
                                
                                all_gates.append([[all_coords[(i, r_c[1])], all_coords[(i, j)]], error, Cr])
                                x_gates.append([[all_coords[(i, r_c[1])], all_coords[(i, j)]], error, Cr])
                            else:           
                                all_gates.append([[all_coords[(i, r_c[1])], all_coords[(i, j)]], after_clifford_depolarization, 1])
                                x_gates.append([[all_coords[(i, r_c[1])], all_coords[(i, j)]], after_clifford_depolarization, 1])

                            if (i, r_c[1]) in x_pairings:
                                x_pairings[(i, r_c[1])].append((i, j))
                            else:
                                x_pairings[(i, r_c[1])] = [(i, j)]

                        else:
                            error = after_clifford_depolarization
                            Cr = 1
                            if crossings:
                                for crossing in crossings:
                                    if crossing[1] == (i, r_c[1]) and crossing[0] == (i, j):
                                        error = after_crossing_depolarization
                                        Cr = crossing[2]

                                all_gates.append([[all_coords[(i, r_c[1])], all_coords[(i, j)]], error, Cr])
                                z_gates.append([[all_coords[(i, r_c[1])], all_coords[(i, j)]], error, Cr])
                            else:           
                                all_gates.append([[all_coords[(i, r_c[1])], all_coords[(i, j)]], after_clifford_depolarization, 1])
                                z_gates.append([[all_coords[(i, r_c[1])], all_coords[(i, j)]], after_clifford_depolarization, 1])

                            if (i, j) in z_pairings:
                                z_pairings[(i, j)].append((i, r_c[1])) 
                            else:
                                z_pairings[(i, j)] = [(i, r_c[1])]

        for check in sorted(x_gates, key = lambda x : x[0][0]):
            append_gate_2(circ_cycle, 'CNOT', check[0], check[1], check[2]) 

        for check in sorted(z_gates, key = lambda x : x[0][1]):
            append_gate_2(circ_cycle, 'CNOT', check[0], check[1], check[2]) 


        append_gate_1(circ_cycle, 'H', [*x_checks.values()],  after_clifford_depolarization)

        append_MR(circ_cycle, [*x_checks.values()] + [*z_checks.values()], before_measure_flip_probability, after_reset_flip_probability)

        return circ_cycle
    

    # head of circuit:
    noisy_cycle = cycle()
    noisy_head_cycle = cycle()

    head = stim.Circuit()

    for coord, index in all_coords.items():
        head.append('QUBIT_COORDS', [index], [coord[0], coord[1]])
    
    # magically noiseless state initialization:
    append_reset(head, [*data_qubits.values()],0 , basis)
    append_reset(head, [*x_checks.values()] + [*z_checks.values()], 0)

    head += noisy_head_cycle

    if basis == 'Z':
        for i, coord in enumerate([*z_checks.keys()]):
            head.append('DETECTOR',
                           [stim.target_rec(-i-1)],
                           )
    
    elif basis == 'X':
        for i, coord in enumerate([*x_checks.keys()]):
            head.append('DETECTOR',
                           [stim.target_rec(-i-1-len([*z_checks.values()]))],
                           )
    # body of the circuit (that we repeat for the specified number of rounds) 
    body = noisy_cycle.copy()

    if basis == 'Z':
        for i, coord in enumerate([*z_checks.keys()]):
            body.append('DETECTOR',
                        [stim.target_rec(-i-1), stim.target_rec(-i-len([*z_checks.values()] + [*x_checks.values()])-1)],
                        )
    
    elif basis == 'X':
        for i, coord in enumerate([*x_checks.keys()]):
            body.append('DETECTOR',
                        [stim.target_rec(-i-1-len([*z_checks.values()])),stim.target_rec(-i-len([*z_checks.values()] + [*z_checks.values()] + [*x_checks.values()])-1)],
                        )

    # tail of the circuit
    tail = stim.Circuit()
    #magically noiseless final data qubit measurements:
    append_M(tail, [*data_qubits.values()], 0, basis)

    reindexed_data_q = {index : i for i, index in enumerate(data_qubits)}

    if basis == 'Z':
        for i, z_pair in enumerate(z_pairings):
            tail.append(
                'DETECTOR',
                [stim.target_rec(-len(data_qubits) + reindexed_data_q[i]) for i in z_pairings[z_pair]] \
                + [stim.target_rec(-(len(data_qubits) + len(z_checks)) + i)]
                        ) 
            
    elif basis == 'X':
        for i, x_pair in enumerate(x_pairings):
            tail.append(
                'DETECTOR',
                [stim.target_rec(-len(data_qubits) + reindexed_data_q[i]) for i in x_pairings[x_pair]] \
                + [stim.target_rec(-(len(data_qubits) + len(z_checks) + len(x_checks)) + i)]
                        ) 
            
    if observable != False :

        if basis == 'Z':

            for obs in observable:
                logical_obs = []

                for j in range(qubit_matrix.shape[0]):
                    if (j, obs) in reindexed_data_q:
                        logical_obs.append(-len(data_qubits) + reindexed_data_q[(j, obs)])

                tail.append('OBSERVABLE_INCLUDE', [stim.target_rec(rec) for rec in logical_obs], obs)
        
        elif basis == 'X':

            for i, obs in enumerate(observable):
                tail.append('OBSERVABLE_INCLUDE', [stim.target_rec(-len(data_qubits) + reindexed_data_q[obs_q]) for obs_q in obs], i)
            
    return head + body * (rounds-1) + tail





    