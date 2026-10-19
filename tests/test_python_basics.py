from grader_contracts.python_basics import TextInput, PositiveIntegerInput, VectorPairInput
"""Примеры и граничные случаи для каждой задачи первой части."""
import unittest
from python_basics import count_vowels, has_unique_characters, count_one_bits, multiplicative_persistence, mse, prime_factorization, pyramid, is_balanced_number

class PythonBasicsTests(unittest.TestCase):

    def test_vowels(self):
        for text, expected in [('', 0), ('AEiou', 5), ('Hello World', 3), ('rhythm', 0), ('aaa', 3)]:
            with self.subTest(text=text):
                self.assertEqual(count_vowels(TextInput(text)), expected)

    def test_unique(self):
        for text, expected in [('', True), ('a', True), ('abc', True), ('aba', False), ('aA', True), ('  ', False), ('ёж', True)]:
            with self.subTest(text=text):
                self.assertIs(has_unique_characters(TextInput(text)), expected)

    def test_bits(self):
        for number, expected in [(0, 0), (1, 1), (2, 1), (7, 3), (42, 3), (2 ** 100, 1), (2 ** 100 - 1, 100)]:
            with self.subTest(number=number):
                self.assertEqual(count_one_bits(PositiveIntegerInput(number)), expected)
        with self.assertRaises(ValueError):
            count_one_bits(PositiveIntegerInput(-1))

    def test_persistence(self):
        for number, expected in [(0, 0), (4, 0), (9, 0), (10, 1), (39, 3), (999, 4), (100, 1), (77, 4)]:
            with self.subTest(number=number):
                self.assertEqual(multiplicative_persistence(PositiveIntegerInput(number)), expected)
        with self.assertRaises(ValueError):
            multiplicative_persistence(PositiveIntegerInput(-1))

    def test_mse(self):
        for x, y, expected in [([1, 2, 3], [1, 2, 3], 0), ([1, 2, 3], [2, 4, 6], 14 / 3), ([-2], [3], 25), ([0, 0], [1, 0], 0.5), ([10 ** 30], [0], 10 ** 60)]:
            with self.subTest(x=x, y=y):
                self.assertAlmostEqual(mse(VectorPairInput(x, y)), expected)
        for x, y in [([], []), ([1], [1, 2])]:
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                mse(VectorPairInput(x, y))

    def test_factorization(self):
        for number, expected in [(1, ''), (2, '(2)'), (13, '(13)'), (64, '(2**6)'), (49, '(7**2)'), (86240, '(2**5)(5)(7**2)(11)'), (30, '(2)(3)(5)')]:
            with self.subTest(number=number):
                self.assertEqual(prime_factorization(PositiveIntegerInput(number)), expected)
        with self.assertRaises(ValueError):
            prime_factorization(PositiveIntegerInput(0))

    def test_pyramid(self):
        for k in [1, 2, 3, 10, 100, 10 ** 6]:
            with self.subTest(k=k):
                n = k * (k + 1) * (2 * k + 1) // 6
                self.assertEqual(pyramid(PositiveIntegerInput(n)), k)
                self.assertEqual(pyramid(PositiveIntegerInput(n + 1)), 'It is impossible')
        for n in [2, 4, 6, 13]:
            with self.subTest(n=n):
                self.assertEqual(pyramid(PositiveIntegerInput(n)), 'It is impossible')
        with self.assertRaises(ValueError):
            pyramid(PositiveIntegerInput(0))

    def test_balanced(self):
        for number, expected in [(0, True), (7, True), (12, True), (121, True), (123, False), (1234006, True), (123456, False), (1221, True), (1231, True), (1232, False)]:
            with self.subTest(number=number):
                self.assertIs(is_balanced_number(PositiveIntegerInput(number)), expected)
        with self.assertRaises(ValueError):
            is_balanced_number(PositiveIntegerInput(-1))
