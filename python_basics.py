"""Заготовки задач на базовый Python."""

from math import isqrt

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum((char in 'aeiou' for char in text.lower()))


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    if number < 0:
        raise ValueError('Число должно быть неотрицательным')
    count = 0
    while number:
        number &= number - 1
        count += 1
    return count


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    if number < 0:
        raise ValueError('Число должно быть неотрицательным')
    steps = 0
    while number >= 10:
        product = 1
        while number:
            number, digit = divmod(number, 10)
            product *= digit
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected) or len(predicted) == 0:
        raise ValueError('Нужны непустые векторы одинаковой длины')
    return sum(((int(a) - int(b)) ** 2 for a, b in zip(predicted, expected))) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number < 1:
        raise ValueError('Число должно быть положительным')
    factors = []
    divisor = 2
    while divisor * divisor <= number:
        exponent = 0
        while number % divisor == 0:
            number //= divisor
            exponent += 1
        if exponent:
            factors.append(f'({divisor}**{exponent})' if exponent > 1 else f'({divisor})')
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        factors.append(f'({number})')
    return ''.join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    if cube_count < 1:
        raise ValueError('Количество кубиков должно быть положительным')
    left, right = (1, isqrt(cube_count))
    while left <= right:
        k = (left + right) // 2
        total = k * (k + 1) * (2 * k + 1) // 6
        if total == cube_count:
            return k
        if total < cube_count:
            left = k + 1
        else:
            right = k - 1
    return 'It is impossible'


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    if number < 0:
        raise ValueError('Число должно быть неотрицательным')
    digits = str(number)
    side_length = (len(digits) - 1) // 2
    if side_length == 0:
        return True
    return sum(map(int, digits[:side_length])) == sum(map(int, digits[-side_length:]))
