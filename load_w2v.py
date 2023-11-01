from utils.prepare_data import load_w2v_semeval

# def main(_):
word_idx = {}
word_idx_rev, word_idx, spe_idx_rev, spe_idx, word_embedding, _ = load_w2v_semeval(300, 50, './text/Subtask_1_train.json', './ECF_glove_300.txt')

    


# if __name__ == '__main__':
#     tf.app.run() 