import numpy as np
import RNA


one_mer_comb_map = {
     'A': 0, 'T': 1, 'C': 2, 'G': 3
}
two_mer_comb_map = {
    'AA': 0, 'AT': 1, 'AC': 2, 'AG': 3,
    'TA': 4, 'TT': 5, 'TC': 6, 'TG': 7,
    'CA': 8, 'CT': 9, 'CC': 10, 'CG': 11,
    'GA': 12, 'GT': 13, 'GC': 14, 'GG': 15
}

def get_CG_freq(dna_sequence):
    # Count the frequency of G and C nucleotides
    gc_count = dna_sequence.count('G') + dna_sequence.count('C')
    gc_frequency = gc_count / len(dna_sequence)
    gc_flag = 0
    if gc_frequency >= 0.4 and gc_frequency <=0.6:
        gc_flag = 1
    return gc_flag

def count_stem_loops(structure):
    stack = []
    loops = 0
    for char in structure:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
                loops += 1
    return loops

def dna_to_onehot(dna_sequences):
    batch_size = len(dna_sequences)
    sequence_length = 60
    num_features = 4 * sequence_length + 2
    
    # Initialize a zero matrix of shape (batch_size, num_features)
    onehot_matrix = np.zeros((batch_size, num_features), dtype=float)
    
    # Define the mapping from nucleotide to column index
    one_mer_comb_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    
    # Iterate over each DNA sequence in the batch
    for batch_index, dna_sequence in enumerate(dna_sequences):
        if len(dna_sequence) > sequence_length:
            dna_sequence = dna_sequence[:sequence_length]
        gRNA = dna_sequence[((sequence_length // 2) - 17):(sequence_length // 2 + 3)]
        gc_flag = get_CG_freq(gRNA)
        rna_sequence = gRNA.replace("T", "U")
        structure, _ = RNA.fold(rna_sequence)
        stem_loop_count = count_stem_loops(structure)
        # Iterate over each nucleotide in the sequence
        for i, nucleotide in enumerate(dna_sequence):
            if nucleotide in one_mer_comb_map:
                col_index = i * 4 + one_mer_comb_map[nucleotide]
                onehot_matrix[batch_index, col_index] = 1
            else:
                raise ValueError(f"Invalid nucleotide found in the sequence: {nucleotide}")
        onehot_matrix[batch_index, -2] = gc_flag
        onehot_matrix[batch_index, -1] = stem_loop_count
        
    return onehot_matrix