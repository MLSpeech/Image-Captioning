from utils import make_input_files

# This class serves one purpose only- defining the different input paths and parameters.
if __name__ == '__main__':
    # Making input files.
    make_input_files(dataset='coco', karpathy_json_path='/home/mlspeech/benmosn4/dataset_coco.json',
                       image_folder='/home/mlspeech/benmosn4/', captions_per_image=5,
                       min_word_freq=5, output_folder='/home/mlspeech/benmosn4/output_folder/',
                       max_len=50)
