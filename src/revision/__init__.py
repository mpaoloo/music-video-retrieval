"""Повторный эксперимент с фиксированным протоколом оценки."""
import os

# Используем PyTorch; необязательные TensorFlow/JAX в Colab могут конфликтовать.
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
