import pickle

from tqdm import tqdm


# To accelerate fetching document from a large corpus
def build_line_offset_index(input_path, output_index_path, estimated_line_count=49182879):
    offsets = []
    with open(input_path, 'rb') as f:
        if estimated_line_count is not None:
            pbar = tqdm(total=estimated_line_count, desc="Building offset index")
        else:
            pbar = tqdm(desc="Building offset index")

        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(offset)
            pbar.update(1)

        pbar.close()

    with open(output_index_path, 'wb') as f:
        pickle.dump(offsets, f)

    print(f"\nIndexed {len(offsets)} lines, saved to {output_index_path}")


if __name__ == '__main__':
    build_line_offset_index("data/enwiki_20231101/corpus.jsonl", "data/enwiki_20231101/corpus_line_offset.pkl")
