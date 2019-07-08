# Image Captioning
This is an Image Captioning implementation in PyTorch. We implement the paper ["Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"](https://arxiv.org/abs/1502.03044)
# Requirements:
- Python 3.6.8
- PyTorch 1.0.1
# MS-COCO Dataset:
- Download the following images:
- train2014: http://images.cocodataset.org/zips/train2014.zip
- val2014: http://images.cocodataset.org/zips/val2014.zip
- Extract and put them in the same directory.
# Run the code and start training:
  - Define the required paths in `make_input_file.py`
  - Run: `python make_input_files.py`
- There are four training files:
  - `train.py` using CrossEntropy loss.
   - `trainL2.py` using L2 loss.
  - `trainL1.py` using L1 loss.
  - `train_cosine.py` using Cosine Similarity loss.
- Train the model with your desired loss function: `python desired_train.py`
- Evaluate the model with: `python evaluate.py`
- Caption a new image with: `python caption_image.py --img='path/to/image.jpeg' --model='path/to/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='path/to/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5`
# References:
The authors' original implementation: https://github.com/kelvinxu/arctic-captions
