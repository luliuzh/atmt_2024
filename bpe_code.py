from subword_nmt import apply_bpe, learn_bpe

def merge_files(source_file_path, target_file_path, output_file_path):
    with open(source_file_path, 'r', encoding='utf-8') as source_file, \
         open(target_file_path, 'r', encoding='utf-8') as target_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        
        for source_line, target_line in zip(source_file, target_file):
            output_file.write(source_line)
            output_file.write(target_line)


def bpe_learn(input_file, output_file):
    with open(input_file, 'r') as train_file, open(output_file, 'w') as codes_file:
        learn_bpe.learn_bpe(train_file, codes_file, num_operations=30000)

merge_files('atmt_2024/data/en-fr/raw/train.en', 'atmt_2024/data/en-fr/raw/train.fr', 'atmt_2024/data/en-fr/raw/train_en_fr.txt')
bpe_learn('atmt_2024/data/en-fr/raw/train_en_fr.txt', 'atmt_2024/data/en-fr/raw/bpe_30000.codes')