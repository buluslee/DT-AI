import torch
import numpy as np
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import corpus_bleu
from nltk import word_tokenize

from config import Config
from process import DataProcessor, load_file
from model import Encoder, Decoder, Seq2Seq, LanguageModelCriterion


def set_seed(seed):
    # set seed for CPU
    torch.manual_seed(seed)
    # set seed for current GPU
    torch.cuda.manual_seed(seed)
    # set seed for all GPU
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # Cancel acceleration
    # torch.backends.cudnn.benchmark = False

    np.random.seed(seed)


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = LanguageModelCriterion()
        self.criterion.to(self.config.device)

    def train(self, train_data, test_data=None):
        LOG_FILE = "translation_model.log"
        t_total = self.config.num_epoch * len(train_data)
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.config.warmup_steps,
                                                    num_training_steps=t_total)
        global_step = 0
        total_num_words = total_loss = 0.
        logg_loss = 0.
        logg_num_words = 0.
        best_val_loss = float("inf")
        for epoch in range(1, self.config.num_epoch + 1):
            for train_step, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(train_data):
                self.model.train()
                # （英文batch，英文长度，中文batch，中文长度）
                mb_x = torch.from_numpy(mb_x).to(self.config.device).long()
                mb_x_len = torch.from_numpy(mb_x_len).to(self.config.device).long()
                # 前n-1个单词作为输入，后n-1个单词作为输出，因为输入的前一个单词要预测后一个单词
                mb_input = torch.from_numpy(mb_y[:, :-1]).to(self.config.device).long()
                mb_output = torch.from_numpy(mb_y[:, 1:]).to(self.config.device).long()
                mb_y_len = torch.from_numpy(mb_y_len - 1).to(self.config.device).long()

                # 输入输出的长度都减一。
                mb_y_len[mb_y_len <= 0] = 1

                mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)
                """
                这里生成mask，比如：
                mb_y_len = tensor([6,7,8])
                tensor([[ True,  True,  True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True,  True,  True,  True]])
                """
                mb_out_mask = torch.arange(mb_y_len.max().item(), device=self.config.device)[None, :] < mb_y_len[:,
                                                                                                        None]
                # batch,seq_len . 其中每行长度超过自身句子长度的为false
                mb_out_mask = mb_out_mask.float()
                mb_out_mask = mb_out_mask.to(self.config.device)
                """
                这里计算损失，比如
                BOS 我 爱 你
                我  爱 你 EOS
                """
                loss = self.criterion(mb_pred, mb_output, mb_out_mask)
                # 损失函数

                # 更新模型
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.GRAD_CLIP)
                # 为了防止梯度过大，设置梯度的阈值
                optimizer.step()
                scheduler.step()

                global_step += 1
                num_words = torch.sum(mb_y_len).item()
                # 一个batch里多少个单词
                total_loss += loss.item() * num_words
                # 总损失，loss计算的是均值损失，每个单词都是都有损失，所以乘以单词数
                total_num_words += num_words
                # 总单词数

                if (global_step + 1) % 100 == 0:
                    loss_scalar = (total_loss - logg_loss) / (total_num_words - logg_num_words)
                    logg_num_words = total_num_words
                    logg_loss = total_loss

                    with open(LOG_FILE, "a") as fout:
                        fout.write(
                            "epoch: {}, iter: {}, loss: {},learn_rate: {}\n".format(epoch, global_step, loss_scalar,
                                                                                    scheduler.get_lr()[0]))
                    print("epoch: {}, iter: {}, loss: {}, learning_rate: {}".format(epoch, global_step, loss_scalar,
                                                                                    scheduler.get_lr()[0]))
                global_step += 1
            print("Epoch", epoch, "Training loss", total_loss / total_num_words)
            if self.config.do_test:
                eval_loss = self.test(test_data)  # 评估模型
                with open(LOG_FILE, "a") as fout:
                    fout.write("===========" * 20)
                    fout.write("EVALUATE: epoch: {}, loss: {}\n".format(epoch, eval_loss))
                if eval_loss < best_val_loss:
                    # 如果比之前的loss要小，就保存模型
                    best_val_loss = eval_loss
                    print("best model, val loss: ", eval_loss)
                    torch.save(model.state_dict(), os.path.join(self.config.save_dir, "translate-best.pt"))

    def test(self, test_data):
        self.model.eval()
        total_num_words = total_loss = 0.
        with torch.no_grad():  # 不需要更新模型，不需要梯度
            for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(test_data):
                mb_x = torch.from_numpy(mb_x).to(self.config.device).long()
                mb_x_len = torch.from_numpy(mb_x_len).to(self.config.device).long()
                mb_input = torch.from_numpy(mb_y[:, :-1]).to(self.config.device).long()
                mb_output = torch.from_numpy(mb_y[:, 1:]).to(self.config.device).long()
                mb_y_len = torch.from_numpy(mb_y_len - 1).to(self.config.device).long()
                mb_y_len[mb_y_len <= 0] = 1

                mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

                mb_out_mask = torch.arange(mb_y_len.max().item(), device=self.config.device)[None, :] < mb_y_len[:,
                                                                                                        None]
                mb_out_mask = mb_out_mask.float()
                mb_out_mask = mb_out_mask.to(self.config.device)
                loss = self.criterion(mb_pred, mb_output, mb_out_mask)

                num_words = torch.sum(mb_y_len).item()
                total_loss += loss.item() * num_words
                total_num_words += num_words
        print("Evaluation loss", total_loss / total_num_words)
        return total_loss / total_num_words

    def evaluate(self):
        self.model.eval()
        en_sents, ch_sents = load_file(self.config.data_dir + 'test.txt', add_begin_end=False)
        en_sents = [[processor.en_tokenizer.word2idx.get(word, config.UNK_IDX) for word in sen] for sen in en_sents]

        top_hypotheses = []
        with torch.no_grad():
            for idx, en_sent in enumerate(en_sents):
                mb_x = torch.from_numpy(np.array(en_sent).reshape(1, -1)).long().to(self.config.device)
                mb_x_len = torch.from_numpy(np.array([len(en_sent)])).long().to(self.config.device)
                bos = torch.Tensor([[processor.ch_tokenizer.word2idx['BOS']]]).long().to(self.config.device)
                completed_hypotheses = self.model.beam_search(mb_x, mb_x_len,
                                                              bos, processor.ch_tokenizer.word2idx['EOS'],
                                                              topk=self.config.beam_size,
                                                              max_length=self.config.max_beam_search_length)
                top_hypotheses.append([processor.ch_tokenizer.idx2word[id] for id in completed_hypotheses[0].value])

        bleu_score = corpus_bleu([[ref] for ref in ch_sents],
                                 top_hypotheses)

        print('Corpus BLEU: {}'.format(bleu_score * 100))
        return bleu_score

    def predict(self, text):
        self.model.eval()
        text = ['BOS'] + word_tokenize(text.lower()) + ['EOS']
        text_id = [processor.en_tokenizer.word2idx.get(word, 1) for word in text]
        mb_x = torch.from_numpy(np.array(text_id).reshape(1, -1)).long().to(device)
        mb_x_len = torch.from_numpy(np.array([len(text_id)])).long().to(device)

        bos = torch.Tensor([[processor.ch_tokenizer.word2idx['BOS']]]).long().to(device)

        completed_hypotheses = model.beam_search(mb_x, mb_x_len,
                                                 bos, processor.ch_tokenizer.word2idx['EOS'],
                                                 topk=self.config.beam_size,
                                                 max_length=self.config.max_beam_search_length)

        for hypothes in completed_hypotheses:
            result = "".join([processor.ch_tokenizer.idx2word[id] for id in hypothes.value])
            score = hypothes.score
            print("翻译后的中文结果为:{},score:{}".format(result, score))


if __name__ == '__main__':
    set_seed(123)
    config = Config()

    # 默认使用第0块gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.device = device

    processor = DataProcessor(config)
    enc_vocab_size = processor.en_tokenizer.vocab_size
    dec_vocab_size = processor.ch_tokenizer.vocab_size
    config.enc_vocab_size = enc_vocab_size
    config.dec_vocab_size = dec_vocab_size

    if config.do_train:
        train_examples = processor.get_train_examples(config)

    if config.do_test:
        test_examples = processor.get_dev_examples(config)

    encoder = Encoder(config)
    decoder = Decoder(config)
    model = Seq2Seq(encoder, decoder)
    model.to(device)

    if config.do_load_model:
        model.load_state_dict(torch.load(config.load_dir))

    trainer = Trainer(model, config)

    if config.do_train:
        if config.do_test:
            trainer.train(train_examples, test_examples)
        else:
            trainer.train(train_examples)

    if config.do_evaluate:
        trainer.evaluate()

    if config.do_predict:
        text = 'how old are you?'
        trainer.predict(text)
