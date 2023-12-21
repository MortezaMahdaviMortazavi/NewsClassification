import unittest
import torch
from models import RNN

class TestRNN(unittest.TestCase):
    def setUp(self):
        vocab_size = 1000
        embed_size = 50
        embed_proj_size = 50
        rnn_hidden_size = 64
        output_size = 10
        is_bidirectional = True
        n_layer = 2
        num_classes = 5
        dropout_rate = 0.2

        self.model = RNN(
            vocab_size=vocab_size,
            embed_size=embed_size,
            embed_proj_size=embed_proj_size,
            rnn_hidden_size=rnn_hidden_size,
            output_size=output_size,
            is_bidirectional=is_bidirectional,
            n_layer=n_layer,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )

    def test_forward(self):
        input_tensor = torch.randint(0, 1000, (2, 10))
        logits = self.model(input_tensor)
        print(logits)
        self.assertEqual(logits.shape, torch.Size([2, 5]))

if __name__ == '__main__':
    unittest.main()
