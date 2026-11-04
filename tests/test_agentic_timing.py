import unittest
from experiments.agentic_timing import parse, summarize, union_seconds


class TimingTests(unittest.TestCase):
    def test_interval_union(self):
        self.assertEqual(union_seconds([(0, 5), (1, 2), (3, 8), (10, 11)]), 9)
        self.assertEqual(union_seconds([]), 0)

    def test_concurrent_rates_do_not_sum_elapsed_time(self):
        text = '\n'.join([
            '0.02.000.000 I slot print_timing: id  0 | task 1 | prompt eval time = 1000.00 ms / 100 tokens',
            '0.02.000.010 I slot print_timing: id  0 | task 1 |        eval time = 1000.00 ms / 50 tokens',
            '0.02.000.000 I slot print_timing: id  1 | task 2 | prompt eval time = 1000.00 ms / 100 tokens',
            '0.02.000.010 I slot print_timing: id  1 | task 2 |        eval time = 1000.00 ms / 50 tokens',
            '0.03.000.000 I slot operator(): id  0 | task 3 | new prompt, task.n_tokens = 1000',
        ])
        rows = parse(text)
        self.assertEqual(len(rows), 2)  # Exclude unfinished requests.
        summary = summarize(rows)
        self.assertEqual(summary['prefill']['combined_tok_s'], 200)
        self.assertEqual(summary['decode']['combined_tok_s'], 100)
        self.assertEqual(summary['decode']['whole_window_tok_s'], 50)
        self.assertEqual(summary['prefill_decode_overlap_s'], 0)
