import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = '/home/mlspeech/benmosn4/output_folder/'  # folder with data files saved by make_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Training parameters
start_epoch = 0
epochs_num = 130  # number of epochs to train for (if early stopping is not triggered).
# Epochs num since there's been an improvement in validation BLEU-4.
epochs_since_improvement = 0
batch_size = 32
workers = 1  # for data-loading. right now, only 1 works with h5py
fine_tune_encoder = False
checkpoint = None  # path to checkpoint, initialized to None.
encoder_lr = 1e-4  # learning rate for encoder.
decoder_lr = 4e-4  # learning rate for decoder.
grad_clip = 5.
c_reg = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.
# Print training/validation stats every 100 batches.
print_freq = 100


# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda")
cudnn.benchmark = True




def main():
    """
    Training and validation.
    """

    global best_bleu4, start_epoch, fine_tune_encoder, data_name, word_map, epochs_since_improvement, checkpoint

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize checkpoint.
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
    # Not the first checkpoint.
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU.
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CosineEmbeddingLoss().to(device)

    # Normalize with known mean and std from ResNet.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Start the training loop.
    for epoch in range(start_epoch, epochs_num):

        # Terminate training after 20 epochs without improvement.
        if epochs_since_improvement == 20:
            break
        # Decay learning rate if there is no improvement for 8 consecutive epochs.
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training.
        train(train_loader=train_loader, encoder=encoder, decoder=decoder, criterion=criterion,
              encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader, encoder=encoder,
                                decoder=decoder, criterion=criterion)

        # Check if there was an improvement with bleu-4 accuracy.
        improved = recent_bleu4 > best_bleu4
        # Update to the best accuracy rate.
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not improved:
            epochs_since_improvement += 1
            print("\nThere was no improvement\n")
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, improved)



def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    Most code is similar to 'train' function.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5_accuracy = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error.
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward propagation.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # The targets are all words after <start> and up to <end>.
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            # Calculate loss
            loss = criterion(scores, targets)
            # Add stochastic attention regularization.
            loss += c_reg * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Top scores and accuracies.
            losses.update(loss.item(), sum(decode_lengths))
            top5_scores = accuracy(scores, targets, 5)
            top5_accuracy.update(top5_scores, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            # Print current status.
            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t' 'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 5 Accuracy {top5_scores.val:.3f} ({top5_scores.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5_scores=top5_accuracy))

            # References
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                # remove <start> and pads
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)

        # Calculate BLEU-1 scores.
        weightsBleu1 = (1.0 / 1.0,)
        bleu1 = corpus_bleu(references, hypotheses, weightsBleu1)
        # print("\nBLEU-1 score at beam size of 1 is %.4f." % (bleu1))

        # Calculate BLEU-2 scores.
        weightsBleu2 = (1.0 / 2.0, 1.0 / 2.0,)
        bleu2 = corpus_bleu(references, hypotheses, weightsBleu2)
        # print("\nBLEU-2 score at beam size of 1 is %.4f." % (bleu2))

        # Calculate BLEU-3 scores.
        weightsBleu3 = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,)
        bleu3 = corpus_bleu(references, hypotheses, weightsBleu3)
        # print("\nBLEU-3 score at beam size of 1 is %.4f." % (bleu3))

        # BLEU-4 is the score we would evaluate our model with.
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print('\n Loss - {loss.avg:.3f}, TOP 5 Accuracy - {top5_scores.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses, top5_scores=top5_accuracy, bleu=bleu4))

    return bleu4

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Training one epoch.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time

    losses = AverageMeter()  # loss (per word decoded)
    top5_accuracy = AverageMeter()

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):

        # Move to GPU.
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward propagation.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # The targets are all words after <start> and up to <end>.
        targets = caps_sorted[:, 1:]

        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        # Calculate loss
        loss = criterion(scores, targets)

        # Add stochastic attention regularization.
        loss += c_reg * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back propagation.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        # Back propagation is that simple.
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)
        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Top scores and accuracies.
        top5_scores = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5_accuracy.update(top5_scores, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        # Print current status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 5 Accuracy {top5_scores.val:.3f} ({top5_scores.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time, loss=losses,
                                                                          top5_scores=top5_accuracy))




if __name__ == '__main__':
    main()