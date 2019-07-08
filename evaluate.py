import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# Parameters
output_folder = '/home/mlspeech/benmosn4/output_folder'  # folder with data files saved by make_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = '/home/mlspeech/benmosn4/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
# word map, ensure it's the same the data was encoded with and the model was trained with.
word_map_file = '/home/mlspeech/benmosn4/output_folder/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set to true only if inputs to model are fixed size.
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map.
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
vocabulary_size = len(word_map)
# Normalize with known mean and std (from known networks).
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(output_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)


    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need:
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="Evaluating at Beam Size " + str(beam_size))):

        k = beam_size

        # Move to GPU device.
        image = image.to(device)

        # Encode
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
            awe, _ = decoder.attention(encoder_out, hidden)
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

            # Add new words to sequences.
            top_k_sequences = torch.cat([top_k_sequences[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1)

            # Sequences that didn't reach <end> (therefore, are incomplete)
            incomplete_sequences = [ind for ind, next_word in enumerate(next_word_indices) if
                                    next_word != word_map['<end>']]
            complete_indices = list(set(range(len(next_word_indices))) - set(incomplete_sequences))

            # Set aside complete sequences
            if len(complete_indices) > 0:
                complete_sequences.extend(top_k_sequences[complete_indices].tolist())
                complete_sequences_scores.extend(top_k_scores[complete_indices])
            k -= len(complete_indices)  # reduce beam length accordingly.

            # Proceed with incomplete sequences.
            # When beam size is zero.
            if k == 0:
                break
            top_k_sequences = top_k_sequences[incomplete_sequences]
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

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in caption if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-1 scores.
    weightsBleu1 = (1.0 / 1.0,)
    bleu1 = corpus_bleu(references, hypotheses, weightsBleu1)
    #print("\nBLEU-1 score at beam size of 1 is %.4f." % (bleu1))

    # Calculate BLEU-2 scores.
    weightsBleu2 = (1.0/2.0, 1.0/2.0,)
    bleu2 = corpus_bleu(references, hypotheses, weightsBleu2)
    #print("\nBLEU-2 score at beam size of 1 is %.4f." % (bleu2))

    # Calculate BLEU-3 scores.
    weightsBleu3 = (1.0/3.0, 1.0/3.0, 1.0/3.0,)
    bleu3 = corpus_bleu(references, hypotheses, weightsBleu3)
    #print("\nBLEU-3 score at beam size of 1 is %.4f." % (bleu3))

    # BLEU-4 is the score we would evaluate our model with.
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score at beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
