import os

dir_inputs = 'inputs'

input_text = 'bible'#'nietzsche' #'shakespeare' 'tolstoy_wp'  #  #'shakespeare' 'nietzsche' 'tolstoy_wp' 'dosto_crime' 'elizabethans' 'wstein_tractatus' 'lit_english' 'orwell_84_mini' 'orwell_1984' 'herbert_dune' 'n_ecce_homo' 'moliere' 

in_file_path = os.path.join(dir_inputs, f'{input_text}.txt')
with open(in_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

words = text.split()

word_count = len(words)
word_count_unique = len(set(words))

print(f'input_text = {input_text}')
print(f'word_count = {word_count}')
print(f'word_count_unique = {word_count_unique}')