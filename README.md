Heavily modified version of Karpathy's mini GPT, as per the notes below. This is pretty poorly documented, as I neglected to document changes as I was making them

main file is now gpt_R_w_toks (GPT, my initial, with tokenisation)

Still does not initialise properly, as I've not made any changes to K's original script in that regard, though I have made several other changes.



Changes (from memory, may not be 100% accurate):

1 Added byte-encoder and decoder for tokenization
     (This, from memory, is similar to something constructed by Karpathy during the lesson, but not the same as what he saved on his github)

2 Now saves model to models directory at end of each training run

3 Saves vocab, merges and encoded text for quick use in future training runs

4 Saves hyperparams for use in future training/inference

5 New generator.py file can be used to import a saved model and run inference on it to generate new outputs

6 Several metrics are saved to a general log file in logs folder. This output is a bit of a mish-mash of much of the above, and needs better formatting



Any inputs you want to use should be stored as .txt files in an "inputs" directory, itself stored in the root folder

"models" and "logs" directories can be created to contain the outputs from the gpt_R_w_toks script. otherwise, these will be created automatically at run-time

"generations" directory can be created to contain the outputs from the generator.py. otherwise, this will be created automatically at run-time

-----

Repeat: This is nowhere near as organised as it could be, in its current form. Use at own risk

-----

Karpathy's original gpt is still saved here as gpt.py, but it's better to get that from Karpathy's github, as you know you're getting the original with zero tinkering from me.

Karpathy's original notes:

# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT
