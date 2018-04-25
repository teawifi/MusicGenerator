import numpy as np
import tensorflow as tf
import os
import glob
from tqdm import tqdm

import midi_manipulation
from rbm import RBM

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs


songs = get_songs('Pop_Music_Midi')
print("{} songs processed".format(len(songs)))

num_timesteps = 50
lowest_note = midi_manipulation.lowerBound
highest_note = midi_manipulation.upperBound
note_range = highest_note-lowest_note

visible_units = 2*note_range*num_timesteps
hidden_units = 50

number_epoches = 100
batch_size = 100
alpha = 0.005

#The placeholder variable that holds our data
x = tf.placeholder(tf.float32, [None, visible_units], name="x")

model = RBM(visible_units, hidden_units)
updated_parameters = model.fit(x, alpha)

with tf.Session() as session:
    logs_dir = './logs'
    mse = 0
    init = tf.global_variables_initializer()
    session.run(init)

    for epoch in tqdm(range(number_epoches)):
        for song in songs:
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0] / num_timesteps) * num_timesteps)]
            song = np.reshape(song, [int(song.shape[0] / num_timesteps), song.shape[1] * num_timesteps])
            for i in range(1, len(song), batch_size):
                train_example = song[i:i + batch_size]
                _, _, _, mse = session.run(updated_parameters, feed_dict={x: train_example})

        print("\nEpoch: ", epoch+1, ", mse = {:.5f}".format(mse))

    saver = tf.train.Saver()
    save_path = os.path.join(logs_dir, 'music_generator_RBM_model_20timesteps_200epoches_50hidden_units_0005alpha.ckpt')
    saver.save(session, save_path)

    probs = session.run(model.forward_propagation(x), feed_dict={x: np.zeros([hidden_units, visible_units],
                                                                             dtype='float32')})
    h = session.run(model.sample(probs))
    new_songs, _ = session.run(model.gibbs_sample(1, h))

    for i in range(new_songs.shape[0]):
        if not any(new_songs[i, :]):
            continue
        new_hit = np.reshape(new_songs[i], [num_timesteps, 2*note_range])
        midi_manipulation.noteStateMatrixToMidi(new_hit, "new_hit_{}".format(i))