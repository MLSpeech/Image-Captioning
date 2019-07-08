import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import argparse
from scipy.misc import imread, imresize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_caption(image_path, caption, reversed_word_map):
    """
    This function prints the caption for the given image.
    :param image_path: path to image that has been captioned.
    :param caption: The caption for the given image.
    :param reversed_word_map: reverse word mapping.
    """
    words = [reversed_word_map[ind] for ind in caption]
    print("\n")
    print(*words)
    print("\n")


def beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    The function reads an image and captions it with beam search.
    Similar code to the evaluate function in evaluate.py.

    :param encoder: encoder model.
    :param decoder: decoder model.
    :param image_path: path to the given image.
    :param word_map: word map.
    :param beam_size: number of sequences to consider at each decode-step when running beam search.
    :return: caption
    """

    k = beam_size
    vocabulary_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # Resize image to 256x256.
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)

    # Normalize with known mean and std (from known networks).
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # insert singleton dim(1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    # Creating a tensor to store the top k previous words at each step
    # now all initialized to <start>
    k_previous_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    # Creating a tensor to store top k sequences.
    # now all initialized to <start>
    top_k_sequences = k_previous_words
    # Creating a tensor to store top k sequences' scores.
    # now all initialized to 0
    top_k_scores = torch.zeros(k, 1).to(device)
    # Creating a tensor to store top k sequences' alphas.
    # now all initialized to 1s.
    sequences_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)
    # Lists to store completed sequences and their scores.
    complete_sequences = list()
    complete_sequences_scores = list()

    # Start decoding (using LSTM)
    step = 1
    hidden, cell_state = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_previous_words).squeeze(1)
        awe, alpha = decoder.attention(encoder_out, hidden)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(hidden))
        awe = gate * awe

        hidden, cell_state = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (hidden, cell_state))

        scores = decoder.fc(hidden)
        scores = F.log_softmax(scores, dim=1)

        # Add the top k scores.
        scores = top_k_scores.expand_as(scores) + scores

        # For the first step, all k points will have the same scores.
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            # Else, find top k scores.
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_indices = top_k_words / vocabulary_size
        next_word_indices = top_k_words % vocabulary_size

        # Add new words to sequences, alphas
        top_k_sequences = torch.cat([top_k_sequences[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1)
        sequences_alpha = torch.cat([sequences_alpha[prev_word_indices], alpha[prev_word_indices].unsqueeze(1)], dim=1)

        # Sequences that didn't reach <end> (therefore, are incomplete)
        incomplete_sequences = [ind for ind, next_word in enumerate(next_word_indices) if
                           next_word != word_map['<end>']]
        complete_indices = list(set(range(len(next_word_indices))) - set(incomplete_sequences))

        # Set aside complete sequences
        if len(complete_indices) > 0:
            complete_sequences.extend(top_k_sequences[complete_indices].tolist())
            complete_sequences_scores.extend(top_k_scores[complete_indices])
        k -= len(complete_indices)  # reduce beam length accordingly

        # Proceed with incomplete sequences.
        # When beam size is zero.
        if k == 0:
            break
        top_k_sequences = top_k_sequences[incomplete_sequences]
        sequences_alpha = sequences_alpha[incomplete_sequences]
        hidden = hidden[prev_word_indices[incomplete_sequences]]
        cell_state = cell_state[prev_word_indices[incomplete_sequences]]
        encoder_out = encoder_out[prev_word_indices[incomplete_sequences]]
        top_k_scores = top_k_scores[incomplete_sequences].unsqueeze(1)
        k_previous_words = next_word_indices[incomplete_sequences].unsqueeze(1)
        # If beam search runs too long, break.
        if step > 50:
            break
        step += 1

    i = complete_sequences_scores.index(max(complete_sequences_scores))
    # The selected caption is the one with the max score out of the k sequences.
    caption = complete_sequences[i]


    return caption



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning Implementation')
    # Different arguments when calling 'python caption_image.py'
    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    args = parser.parse_args()

    # Load model (encoder and decoder).
    checkpoint = torch.load(args.model)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map.
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    # Reversing the map.
    reversed_word_map = {v: k for k, v in word_map.items()}

    # Encode and decode with attention and beam search.
    caption = beam_search(encoder, decoder, args.img, word_map, args.beam_size)

    # print caption of best sequence
    print_caption(args.img, caption,reversed_word_map)
